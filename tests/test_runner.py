"""runner: capability dispatch, escalation policy, failure recording.

These tests moved from test_workers.py when the escalation loop left
workers._invoke_with_escalation for runner.invoke_with_escalation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from reindex import config, failures, runner, workers
from reindex.models import LeafSummary
from reindex.providers.base import InvokeResult, Provider, ProviderFailure
from reindex.workers import prepare_leaf

ESCALATION_MODEL = "claude-sonnet-4-6[1m]"  # claude-cli built-in default


class FakeProvider(Provider):
    """Scriptable provider: invoke delegates to an AsyncMock so tests
    control successes/failures per call."""

    name = "fake"

    def __init__(self, invoke_mock: AsyncMock):
        super().__init__(config.load(Path("/nonexistent"), provider_name="claude-cli"))
        self.invoke_mock = invoke_mock

    async def invoke(self, req):
        return await self.invoke_mock(req)


def _ok(payload, cost: float = 0.1, turns: int = 1, duration_ms: int = 100) -> InvokeResult:
    return InvokeResult(payload=payload, cost=cost, turns=turns, duration_ms=duration_ms)


def _result_parse_failure() -> ProviderFailure:
    return ProviderFailure(
        "result is not JSON", kind="result_parse", provider="fake",
        retryable=True, exit_code=0, stdout="some prose the model wrote",
    )


def _process_exit_failure() -> ProviderFailure:
    return ProviderFailure(
        "claude -p exit 1", kind="process_exit", provider="fake",
        retryable=False, exit_code=1,
    )


async def _run_leaf(item, mock: AsyncMock) -> FakeProvider:
    provider = FakeProvider(mock)
    await runner.run_step(
        provider, step="leaf", items=[item],
        schema_cls=LeafSummary,
        finalize_fn=workers.finalize_leaf,
        batch_size=1,
    )
    return provider


# ---------------------------------------------------------------------------
# Failure recording
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unexpected_error_records_failure(tmp_export, make_conv):
    conv = make_conv("foo", "content")
    item = prepare_leaf(conv)
    await _run_leaf(item, AsyncMock(side_effect=RuntimeError("boom")))
    assert failures.count() == 1


# ---------------------------------------------------------------------------
# Escalation retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_leaf_retries_with_escalation_model_on_result_parse(
    tmp_export, make_conv, valid_leaf_dict,
):
    """First call fails retryable on the primary model; second call succeeds
    on the configured escalation model. finalize_leaf should see the
    escalation model + retry=True."""
    conv = make_conv("retry-me", "content")
    item = prepare_leaf(conv)
    assert item is not None  # cache miss path

    good_payload = LeafSummary.model_validate(valid_leaf_dict)
    mock = AsyncMock(side_effect=[
        _result_parse_failure(),
        _ok(good_payload, cost=0.45, turns=7, duration_ms=4000),
    ])
    await _run_leaf(item, mock)

    assert mock.await_count == 2
    second_req = mock.await_args_list[1].args[0]
    assert second_req.model == ESCALATION_MODEL
    # INDEX.md stamped with the escalation model + retry recorded in cost log.
    idx = (conv / "INDEX.md").read_text(encoding="utf-8")
    assert ESCALATION_MODEL in idx
    cost_lines = (tmp_export / ".reindex-costs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(cost_lines[-1])
    assert rec["retry"] is True
    assert rec["model"] == ESCALATION_MODEL


@pytest.mark.asyncio
async def test_leaf_does_not_retry_on_non_retryable_kind(tmp_export, make_conv):
    """retryable=False -- record failure, no second call."""
    conv = make_conv("noretry", "content")
    item = prepare_leaf(conv)

    mock = AsyncMock(side_effect=_process_exit_failure())
    await _run_leaf(item, mock)

    assert mock.await_count == 1
    assert failures.count() == 1


@pytest.mark.asyncio
async def test_leaf_retry_also_fails_records_second_failure(tmp_export, make_conv):
    """Retry fires but escalation model also returns a retryable failure --
    record the second failure, do NOT chain a third attempt."""
    conv = make_conv("doomed", "content")
    item = prepare_leaf(conv)

    mock = AsyncMock(side_effect=[_result_parse_failure(), _result_parse_failure()])
    await _run_leaf(item, mock)

    assert mock.await_count == 2
    assert failures.count() == 1


@pytest.mark.asyncio
async def test_leaf_primary_success_no_retry(tmp_export, make_conv, valid_leaf_dict):
    """Sunny path: primary succeeds, no retry, retry=False in cost log."""
    conv = make_conv("happy", "content")
    item = prepare_leaf(conv)

    good_payload = LeafSummary.model_validate(valid_leaf_dict)
    mock = AsyncMock(return_value=_ok(good_payload))
    await _run_leaf(item, mock)

    assert mock.await_count == 1
    cost_lines = (tmp_export / ".reindex-costs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(cost_lines[-1])
    assert rec["retry"] is False


@pytest.mark.asyncio
async def test_no_escalation_when_config_has_none(tmp_export, make_conv):
    """A provider whose tiers have escalation=None never retries."""
    conv = make_conv("no-esc", "content")
    item = prepare_leaf(conv)

    mock = AsyncMock(side_effect=_result_parse_failure())
    provider = FakeProvider(mock)
    provider.config = config.load(Path("/nonexistent"), provider_name="ollama")
    await runner.run_step(
        provider, step="leaf", items=[item],
        schema_cls=LeafSummary, finalize_fn=workers.finalize_leaf, batch_size=1,
    )
    assert mock.await_count == 1
    assert failures.count() == 1


@pytest.mark.asyncio
async def test_escalate_false_disables_retry(tmp_export, make_conv):
    """run_step(escalate=False) (the root step) never retries even on a
    retryable failure with an escalation model configured."""
    conv = make_conv("root-ish", "content")
    item = prepare_leaf(conv)

    mock = AsyncMock(side_effect=_result_parse_failure())
    provider = FakeProvider(mock)
    await runner.run_step(
        provider, step="leaf", items=[item],
        schema_cls=LeafSummary, finalize_fn=workers.finalize_leaf,
        batch_size=1, escalate=False,
    )
    assert mock.await_count == 1
    assert failures.count() == 1


# ---------------------------------------------------------------------------
# finalize_fn error isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_finalize_error_is_recorded_not_raised(tmp_export, make_conv, valid_leaf_dict):
    """A finalize_fn that raises must NOT propagate out of _invoke_one.

    With gather(return_exceptions=False), an uncaught finalize error would
    cancel all sibling in-flight tasks. The fix wraps finalize_fn in its own
    try/except so one item's disk-full / stamp error is isolated to that item.
    """
    conv = make_conv("finalize-boom", "content")
    item = prepare_leaf(conv)
    assert item is not None

    good_payload = LeafSummary.model_validate(valid_leaf_dict)
    mock = AsyncMock(return_value=_ok(good_payload))

    def bad_finalize(*args, **kwargs):
        raise OSError("disk full")

    provider = FakeProvider(mock)
    # Must not raise — the gather would propagate the exception and kill siblings.
    await runner.run_step(
        provider, step="leaf", items=[item],
        schema_cls=LeafSummary, finalize_fn=bad_finalize, batch_size=1,
    )

    # Invoke succeeded (one call), finalize failure recorded as a failure entry.
    assert mock.await_count == 1
    assert failures.count() == 1


@pytest.mark.asyncio
async def test_finalize_error_does_not_cancel_sibling_tasks(
    tmp_export, make_conv, valid_leaf_dict,
):
    """When one item's finalize fails, sibling items in the same gather
    must still complete successfully.

    Regression guard: before the fix, gather(return_exceptions=False) would
    cancel all pending tasks on the first uncaught exception from finalize_fn.
    """
    convs = [make_conv(f"sib-{i}", "content") for i in range(3)]
    items = [prepare_leaf(c) for c in convs]
    assert all(it is not None for it in items)

    good_payload = LeafSummary.model_validate(valid_leaf_dict)
    mock = AsyncMock(return_value=_ok(good_payload))

    call_count = 0

    def first_item_fails(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("disk full on first item only")

    provider = FakeProvider(mock)
    await runner.run_step(
        provider, step="leaf", items=items,
        schema_cls=LeafSummary, finalize_fn=first_item_fails,
        batch_size=3,
    )

    # All three invokes should have completed.
    assert mock.await_count == 3
    # Only one finalize failure recorded.
    assert failures.count() == 1
