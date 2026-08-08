"""
anthropic provider: direct Anthropic API (key-billed, not subscription).

Two surfaces:
  invoke()      single messages.create with forced tool_use — same request
                shape batch.py builds, full price.
  BatchCapable  Message Batches API at 50% discount; delegates to batch.py,
                which stays the Anthropic-internal batch engine (deliberately
                NOT generalized — extract a neutral engine only when a second
                batch provider exists).

Owns the shared AsyncAnthropic client (moved from backend.py).
"""

from __future__ import annotations

import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from anthropic import AsyncAnthropic, NotFoundError
from pydantic import BaseModel, ValidationError

from reindex import batch, log
from reindex.providers.base import (
    BatchCapable,
    BatchStatus,
    InvokeRequest,
    InvokeResult,
    Provider,
    ProviderFailure,
    validate_or_coerce,
)

_async_client: AsyncAnthropic | None = None


def get_async_client() -> AsyncAnthropic:
    """Shared httpx-pooled AsyncAnthropic client."""
    global _async_client
    if _async_client is None:
        _async_client = AsyncAnthropic()
    return _async_client


async def aclose() -> None:
    global _async_client
    if _async_client is not None:
        await _async_client.close()
        _async_client = None


class AnthropicApiProvider(Provider, BatchCapable):
    name = "anthropic"

    async def preflight(self) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderFailure(
                "ANTHROPIC_API_KEY not set (put it in .env or use --provider claude-cli)",
                kind="process_exit",
                provider=self.name,
            )

    async def aclose(self) -> None:
        await aclose()

    # -- single call ---------------------------------------------------------

    async def invoke(self, req: InvokeRequest) -> InvokeResult:
        """One messages.create with the same forced-tool_use request the
        batch path submits, minus the batch discount."""
        blog = log.get("provider.anthropic").bind(step=req.step, slug=req.slug)
        schema = req.schema_cls.model_json_schema()
        request = batch._build_request(
            batch.BatchTask(
                custom_id=req.slug or req.step,
                system_prompt=req.system_prompt,
                user_content=req.user_content,
            ),
            step=req.step, model=req.model, schema=schema,
        )

        t0 = time.monotonic()
        try:
            message = await get_async_client().messages.create(**request["params"])
        except Exception as e:
            blog.error("anthropic_http_error", error=str(e)[:300])
            raise ProviderFailure(
                f"anthropic messages.create failed: {e}",
                kind="http_error",
                provider=self.name,
            ) from e
        duration_ms = int((time.monotonic() - t0) * 1000)

        tool_input = None
        for block in message.content:
            if block.type == "tool_use":
                tool_input = block.input
                break
        if tool_input is None:
            raise ProviderFailure(
                "response contained no tool_use block",
                kind="no_structured_output",
                provider=self.name,
                retryable=True,
            )

        try:
            payload = validate_or_coerce(tool_input, req.schema_cls, log_bind=blog, custom_id=req.slug)
        except ValidationError as e:
            raise ProviderFailure(
                f"schema violation: {e}",
                kind="schema_violation",
                provider=self.name,
                retryable=True,
            ) from e

        in_tok = message.usage.input_tokens
        out_tok = message.usage.output_tokens
        return InvokeResult(
            payload=payload,
            cost=self.config.compute_cost(req.model, in_tok, out_tok),
            duration_ms=duration_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    # -- batch capability ----------------------------------------------------

    async def run_batches(
        self,
        *,
        step: str,
        model: str,
        schema_cls: type[BaseModel],
        tasks: list,
        batch_size: int,
        state: Any,
        finalize: Callable[..., Awaitable[None]],
        serialize_context: Callable[[Any], dict],
        max_in_flight: int = 4,
        wait: bool = False,
    ) -> None:
        await batch.run(
            step=step, model=model, schema_cls=schema_cls,
            tasks=tasks, batch_size=batch_size,
            client=get_async_client(),
            finalize=finalize,
            serialize_context=serialize_context,
            state=state,
            max_in_flight=max_in_flight,
            wait=wait,
        )

    async def resume_pending(
        self,
        *,
        state: Any,
        schema_for_step: Callable[[str], type[BaseModel]],
        finalize_persisted: Callable[..., Awaitable[None]],
        max_in_flight: int = 4,
    ) -> None:
        await batch.resume_pending(
            state=state,
            client=get_async_client(),
            schema_for_step=schema_for_step,
            finalize_persisted=finalize_persisted,
            max_in_flight=max_in_flight,
        )

    async def batch_status(self, batch_id: str) -> BatchStatus:
        b = await get_async_client().messages.batches.retrieve(batch_id)
        c = b.request_counts
        return BatchStatus(
            status=b.processing_status,
            done=b.processing_status == "ended",
            counts={
                "succeeded": c.succeeded,
                "errored": c.errored,
                "processing": c.processing,
                "expired": c.expired,
                "canceled": c.canceled,
            },
            expires_at=str(getattr(b, "expires_at", "") or ""),
        )

    async def batch_cancel(self, batch_id: str) -> str:
        b = await get_async_client().messages.batches.cancel(batch_id)
        return b.processing_status

    async def batch_exists(self, batch_id: str) -> bool:
        try:
            await get_async_client().messages.batches.retrieve(batch_id)
            return True
        except NotFoundError:
            return False
