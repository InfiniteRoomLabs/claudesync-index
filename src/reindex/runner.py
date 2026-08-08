"""
Provider-agnostic step execution.

Owns the two policies that used to be smeared across cli.py and workers.py:

  * capability dispatch — `isinstance(provider, BatchCapable)` decides
    between the server-side batch path and the per-item invoke loop.
    This is the ONLY place that branches on provider capability.
  * escalation — retry a failed invoke once on the configured stronger
    model when the provider judged the failure retryable
    (ProviderFailure.retryable). The escalation model comes from
    provider.config.models.escalation; None disables.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel

from reindex import batch, failures, log, paths
from reindex.providers.base import (
    BatchCapable,
    InvokeRequest,
    InvokeResult,
    Provider,
    ProviderFailure,
)
from reindex.state import BatchState

# ---------------------------------------------------------------------------
# Escalation policy (provider-agnostic; was workers._invoke_with_escalation)
# ---------------------------------------------------------------------------

async def invoke_with_escalation(
    provider: Provider,
    req: InvokeRequest,
    *,
    log_bind,
    escalate: bool = True,
) -> tuple[InvokeResult, str, bool]:
    """Run provider.invoke. On a retryable ProviderFailure, retry once
    against the configured escalation model.

    Returns (result, model_actually_used, was_retry). Re-raises the second
    failure unchanged if escalation also fails -- caller records it.
    """
    escalation = provider.config.models.escalation
    try:
        result = await provider.invoke(req)
        return result, req.model, False
    except ProviderFailure as first_err:
        if not escalate or not first_err.retryable or not escalation:
            raise
        log_bind.warning(
            "retrying_with_escalation",
            first_kind=first_err.kind,
            first_model=req.model,
            escalation=escalation,
        )
        result = await provider.invoke(replace(req, model=escalation))
        return result, escalation, True


def record_provider_failure(step: str, slug: str, log_bind, exc: BaseException) -> None:
    """Shared failure-recording path: pull structured context off
    ProviderFailure when present, otherwise fall back to plain str(exc).
    """
    if isinstance(exc, ProviderFailure):
        log_bind.error(
            "backend_failed",
            error=str(exc)[:300],
            kind=exc.kind,
            exit_code=exc.exit_code,
            stderr_preview=(exc.stderr or "")[:500],
        )
        failures.record(
            step=step, slug=slug, kind="backend_failed",
            detail=str(exc), context=exc.to_context(),
        )
    else:
        log_bind.error("backend_failed", error=str(exc)[:300])
        failures.record(step=step, slug=slug, kind="backend_failed", detail=str(exc))


# ---------------------------------------------------------------------------
# Item adapters — the three WorkItem shapes differ in slug / work_dir
# ---------------------------------------------------------------------------

def _slug(item) -> str:
    return getattr(item, "slug", "root")


def _work_dir(item) -> Path:
    for attr in ("conv_dir", "project_dir"):
        d = getattr(item, attr, None)
        if d is not None:
            return d
    return paths.EXPORT_ROOT


def _allow_filesystem(item) -> bool:
    mode = getattr(item, "transcript_mode", None)
    return getattr(mode, "value", None) == "file"


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------

async def run_step(
    provider: Provider,
    *,
    step: str,
    items: list,
    schema_cls: type[BaseModel],
    finalize_fn: Callable[..., None],
    batch_size: int,
    group_by_model: bool = False,
    max_in_flight: int = 4,
    wait: bool = False,
    escalate: bool = True,
) -> None:
    """Execute one step's prepared items on `provider`.

    BatchCapable providers get the server-side batch path (chunked submit,
    state persistence, resumability). Everyone else gets a bounded
    per-item invoke loop with escalation. `batch_size` means "items per
    API batch" on the first path and "max concurrent invokes" on the
    second — same semantics the --batch-size flag always had.
    """
    if not items:
        return

    if isinstance(provider, BatchCapable):
        await _run_batched(
            provider, step=step, items=items, schema_cls=schema_cls,
            finalize_fn=finalize_fn, batch_size=batch_size,
            group_by_model=group_by_model,
            max_in_flight=max_in_flight, wait=wait,
        )
        return

    sem = asyncio.Semaphore(batch_size)

    async def _one(item) -> None:
        async with sem:
            await _invoke_one(
                provider, step=step, item=item, schema_cls=schema_cls,
                finalize_fn=finalize_fn, escalate=escalate,
            )

    await asyncio.gather(*(asyncio.create_task(_one(it)) for it in items),
                         return_exceptions=False)


async def _invoke_one(
    provider: Provider,
    *,
    step: str,
    item,
    schema_cls: type[BaseModel],
    finalize_fn: Callable[..., None],
    escalate: bool,
) -> None:
    slug = _slug(item)
    wlog = log.get("worker").bind(slug=slug)
    wlog.info("started", model=item.model, provider=provider.name, step=step)
    req = InvokeRequest(
        step=step,
        slug=slug,
        model=item.model,
        system_prompt=item.system_prompt,
        user_content=item.user_content,
        schema_cls=schema_cls,
        work_dir=_work_dir(item),
        allow_filesystem=_allow_filesystem(item),
    )
    try:
        result, used_model, was_retry = await invoke_with_escalation(
            provider, req, log_bind=wlog, escalate=escalate,
        )
    except Exception as e:
        record_provider_failure(step, slug, wlog, e)
        return
    try:
        finalize_fn(
            item, result.payload, result.cost, result.turns, result.duration_ms,
            model=used_model, retry=was_retry,
        )
    except Exception as e:
        wlog.error("finalize_failed", error=str(e)[:300])
        failures.record(step=step, slug=slug, kind="finalize_failed", detail=str(e))


# ---------------------------------------------------------------------------
# Batch path glue (moved from cli._api_run_step and friends)
# ---------------------------------------------------------------------------

def _items_to_batch_tasks(items) -> list[batch.BatchTask]:
    return [
        batch.BatchTask(
            custom_id=_slug(it),
            system_prompt=it.system_prompt,
            user_content=it.user_content,
            context=it,
        )
        for it in items
    ]


def _make_finalizer(finalize_fn) -> Callable[..., Awaitable[None]]:
    async def _f(task: batch.BatchTask, payload: BaseModel, cost: float, turns: int, duration_ms: int):
        finalize_fn(task.context, payload, cost, turns, duration_ms)
    return _f


def _serialize_context(item) -> dict:
    from reindex import workers

    serializers = {
        workers.LeafItem: workers.serialize_leaf_item,
        workers.ProjectItem: workers.serialize_project_item,
        workers.RootItem: workers.serialize_root_item,
    }
    return serializers[type(item)](item)


async def _run_batched(
    provider: BatchCapable,
    *,
    step: str,
    items: list,
    schema_cls: type[BaseModel],
    finalize_fn,
    batch_size: int,
    group_by_model: bool,
    max_in_flight: int,
    wait: bool,
) -> None:
    """Submit items as one or more batches. Leaves are grouped by model
    (size-tiered models) so each batch has a uniform model."""
    state = BatchState(paths.EXPORT_ROOT)

    groups: dict[str, list]
    if group_by_model:
        groups = {}
        for it in items:
            groups.setdefault(it.model, []).append(it)
    else:
        groups = {items[0].model: items}

    for model, group in groups.items():
        await provider.run_batches(
            step=step, model=model, schema_cls=schema_cls,
            tasks=_items_to_batch_tasks(group),
            batch_size=batch_size,
            state=state,
            finalize=_make_finalizer(finalize_fn),
            serialize_context=_serialize_context,
            max_in_flight=max_in_flight,
            wait=wait,
        )
