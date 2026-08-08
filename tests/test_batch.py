"""Batch path: submit, poll, retrieve, retry-on-validation, resume."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from reindex import batch, models
from reindex.state import BatchState, PersistedItem

# ---------------------------------------------------------------------------
# Fake Anthropic batch surface
# ---------------------------------------------------------------------------

class _FakeBatch:
    """Mimics anthropic batch retrieve() return."""
    def __init__(self, processing_status: str = "ended", succeeded: int = 1, errored: int = 0):
        self.id = "msgbatch_test"
        self.processing_status = processing_status
        self.request_counts = SimpleNamespace(
            processing=0, succeeded=succeeded, errored=errored,
            canceled=0, expired=0,
        )


class _FakeMessage:
    def __init__(self, content_blocks: list, in_tok: int = 100, out_tok: int = 50):
        self.content = content_blocks
        self.usage = SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok)


class _FakeContentBlock:
    def __init__(self, type_: str, **attrs):
        self.type = type_
        for k, v in attrs.items():
            setattr(self, k, v)


def _success_result(custom_id: str, payload: dict) -> SimpleNamespace:
    """Mimics one MessageBatchIndividualResponse with a tool_use block."""
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="succeeded",
            message=_FakeMessage(
                content_blocks=[_FakeContentBlock("tool_use", id="t_1", input=payload)],
            ),
        ),
    )


def _errored_result(custom_id: str, err: str = "rate_limit") -> SimpleNamespace:
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="errored", error=err),
    )


class _FakeAsyncIter:
    def __init__(self, items):
        self._items = list(items)
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._items):
            raise StopAsyncIteration
        x = self._items[self._i]
        self._i += 1
        return x


def make_fake_client(*, results, batch_status: str = "ended"):
    """Build a MagicMock that mimics AsyncAnthropic surface used by batch.py."""
    client = MagicMock()
    client.messages.batches.create = AsyncMock(return_value=SimpleNamespace(id="msgbatch_test"))
    client.messages.batches.retrieve = AsyncMock(
        return_value=_FakeBatch(processing_status=batch_status, succeeded=len(results)),
    )
    # batches.results is `await client.messages.batches.results(id)` returning iterator.
    client.messages.batches.results = AsyncMock(return_value=_FakeAsyncIter(results))
    return client


# ---------------------------------------------------------------------------
# batch.run happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_single_chunk_all_succeed(valid_leaf_dict, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(batch.asyncio, "sleep", AsyncMock())
    finalize = AsyncMock()

    tasks = [
        batch.BatchTask(custom_id="a", system_prompt="sys", user_content="u-a", context={"slug": "a"}),
        batch.BatchTask(custom_id="b", system_prompt="sys", user_content="u-b", context={"slug": "b"}),
    ]
    results = [
        _success_result("a", valid_leaf_dict),
        _success_result("b", valid_leaf_dict),
    ]
    client = make_fake_client(results=results)
    state = BatchState(tmp_path)

    await batch.run(
        step="leaf", model="claude-haiku-4-5",
        schema_cls=models.LeafSummary,
        tasks=tasks, batch_size=10,
        client=client, finalize=finalize,
        serialize_context=lambda ctx: ctx,
        state=state,
        wait=True,
    )

    assert finalize.await_count == 2
    # Cost discount applied (50%).
    args, _ = finalize.call_args_list[0]
    assert args[2] > 0  # cost
    # State cleared after success.
    assert state.is_empty()


@pytest.mark.asyncio
async def test_run_chunks_into_batches(valid_leaf_dict, tmp_path: Path, monkeypatch):
    """4 tasks with batch_size=2 → 2 batches submitted."""
    monkeypatch.setattr(batch.asyncio, "sleep", AsyncMock())
    finalize = AsyncMock()

    tasks = [
        batch.BatchTask(custom_id=f"id{i}", system_prompt="s", user_content=f"u{i}", context={"slug": f"id{i}"})
        for i in range(4)
    ]
    # results() is called per chunk; provide identical successful payloads each call.
    client = make_fake_client(results=[_success_result(t.custom_id, valid_leaf_dict) for t in tasks[:2]])
    # Replace results to be a fresh iterator each call.
    call_count = {"n": 0}
    def results_per_chunk(_id):
        call_count["n"] += 1
        # Each chunk gets 2 items.
        chunk = tasks[(call_count["n"] - 1) * 2 : call_count["n"] * 2]
        return _FakeAsyncIter([_success_result(t.custom_id, valid_leaf_dict) for t in chunk])
    client.messages.batches.results = AsyncMock(side_effect=results_per_chunk)

    await batch.run(
        step="leaf", model="claude-haiku-4-5",
        schema_cls=models.LeafSummary,
        tasks=tasks, batch_size=2,
        client=client, finalize=finalize,
        serialize_context=lambda c: c,
        state=BatchState(tmp_path),
        wait=True,
    )
    assert client.messages.batches.create.await_count == 2
    assert finalize.await_count == 4


# ---------------------------------------------------------------------------
# Validation failure → auto-retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_retries_validation_failures(valid_leaf_dict, tmp_path: Path, monkeypatch):
    """First batch returns malformed payload (string-instead-of-array bug);
    retry batch returns valid payload. Item should still finalize."""
    monkeypatch.setattr(batch.asyncio, "sleep", AsyncMock())
    finalize = AsyncMock()

    # Concept requires {name, brief}; missing 'brief' is something coerce
    # can't repair (it never invents required content).
    bad_payload = dict(valid_leaf_dict)
    bad_payload["concepts_introduced"] = [{"name": "missing-brief"}]

    tasks = [batch.BatchTask(custom_id="a", system_prompt="s", user_content="u", context={"slug": "a"})]

    call_count = {"n": 0}
    def results_per_call(_id):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeAsyncIter([_success_result("a", bad_payload)])
        else:
            return _FakeAsyncIter([_success_result("a", valid_leaf_dict)])

    client = MagicMock()
    client.messages.batches.create = AsyncMock(return_value=SimpleNamespace(id="b"))
    client.messages.batches.retrieve = AsyncMock(return_value=_FakeBatch())
    client.messages.batches.results = AsyncMock(side_effect=results_per_call)

    await batch.run(
        step="leaf", model="claude-haiku-4-5",
        schema_cls=models.LeafSummary,
        tasks=tasks, batch_size=10,
        client=client, finalize=finalize,
        serialize_context=lambda c: c,
        state=BatchState(tmp_path),
        wait=True,
    )

    # 2 batches submitted (initial + retry).
    assert client.messages.batches.create.await_count == 2
    # Final finalize called once (retry success).
    assert finalize.await_count == 1
    args, _ = finalize.call_args
    # turns=2 indicates retry path.
    assert args[3] == 2


@pytest.mark.asyncio
async def test_run_records_permanent_failure(valid_leaf_dict, tmp_path: Path, monkeypatch):
    """Both initial and retry fail validation → record in failures.jsonl."""
    monkeypatch.setattr(batch.asyncio, "sleep", AsyncMock())
    monkeypatch.setenv("CSINDEX_FAILURE_LOG", str(tmp_path / "failures.jsonl"))
    from reindex import failures
    failures.truncate()
    finalize = AsyncMock()

    bad = dict(valid_leaf_dict)
    # Missing required nested field — coerce can't synthesize content.
    bad["concepts_introduced"] = [{"name": "no-brief"}]

    tasks = [batch.BatchTask(custom_id="a", system_prompt="s", user_content="u", context={"slug": "a"})]

    client = MagicMock()
    client.messages.batches.create = AsyncMock(return_value=SimpleNamespace(id="b"))
    client.messages.batches.retrieve = AsyncMock(return_value=_FakeBatch())
    client.messages.batches.results = AsyncMock(side_effect=lambda _id: _FakeAsyncIter([_success_result("a", bad)]))

    await batch.run(
        step="leaf", model="claude-haiku-4-5",
        schema_cls=models.LeafSummary,
        tasks=tasks, batch_size=10,
        client=client, finalize=finalize,
        serialize_context=lambda c: c,
        state=BatchState(tmp_path),
        wait=True,
    )

    assert finalize.await_count == 0
    assert failures.count() == 1


@pytest.mark.asyncio
async def test_run_handles_errored_result(tmp_path: Path, monkeypatch):
    """Anthropic-side error (rate_limit, expired, etc.) → no finalize, item dropped."""
    monkeypatch.setattr(batch.asyncio, "sleep", AsyncMock())
    finalize = AsyncMock()

    tasks = [batch.BatchTask(custom_id="a", system_prompt="s", user_content="u", context={"slug": "a"})]
    client = MagicMock()
    client.messages.batches.create = AsyncMock(return_value=SimpleNamespace(id="b"))
    client.messages.batches.retrieve = AsyncMock(return_value=_FakeBatch())
    client.messages.batches.results = AsyncMock(return_value=_FakeAsyncIter([_errored_result("a", "rate_limit")]))

    await batch.run(
        step="leaf", model="claude-haiku-4-5",
        schema_cls=models.LeafSummary,
        tasks=tasks, batch_size=10,
        client=client, finalize=finalize,
        serialize_context=lambda c: c,
        state=BatchState(tmp_path),
        wait=True,
    )
    # No success finalizer; retry attempted because outcome had no payload.
    # We expect one extra create() call for the retry.
    assert client.messages.batches.create.await_count == 2


@pytest.mark.asyncio
async def test_run_empty_tasks_noop(tmp_path: Path):
    finalize = AsyncMock()
    client = MagicMock()
    client.messages.batches.create = AsyncMock()
    await batch.run(
        step="leaf", model="m",
        schema_cls=models.LeafSummary,
        tasks=[], batch_size=10,
        client=client, finalize=finalize,
        serialize_context=lambda c: c,
        state=None,
        wait=True,
    )
    assert client.messages.batches.create.await_count == 0
    assert finalize.await_count == 0


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resume_pending_processes_persisted_batches(valid_leaf_dict, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(batch.asyncio, "sleep", AsyncMock())
    state = BatchState(tmp_path)
    state.add(
        batch_id="msgbatch_resumed",
        step="leaf",
        model="claude-haiku-4-5",
        items=[PersistedItem(custom_id="a", step_kwargs={"slug": "a", "model": "claude-haiku-4-5"})],
    )

    client = MagicMock()
    client.messages.batches.retrieve = AsyncMock(return_value=_FakeBatch())
    client.messages.batches.results = AsyncMock(return_value=_FakeAsyncIter([_success_result("a", valid_leaf_dict)]))

    finalize_persisted = AsyncMock()

    await batch.resume_pending(
        state=state,
        client=client,
        schema_for_step=lambda s: models.STEP_MODEL[s],
        finalize_persisted=finalize_persisted,
    )

    assert finalize_persisted.await_count == 1
    # First arg is step name.
    args, _ = finalize_persisted.call_args
    assert args[0] == "leaf"
    assert state.is_empty()


@pytest.mark.asyncio
async def test_resume_pending_records_validation_failures(tmp_path: Path, monkeypatch):
    """Regression: resume_pending must call failures.record() so post-run
    bookkeeping (and exit-code TEMPFAIL escalation) sees the loss."""
    from reindex import failures

    monkeypatch.setattr(batch.asyncio, "sleep", AsyncMock())
    monkeypatch.setenv("CSINDEX_FAILURE_LOG", str(tmp_path / ".reindex-failures.jsonl"))
    failures.truncate()

    state = BatchState(tmp_path)
    state.add(
        batch_id="msgbatch_partial",
        step="leaf",
        model="claude-haiku-4-5",
        items=[
            PersistedItem(custom_id="bad", step_kwargs={"slug": "bad", "model": "claude-haiku-4-5"}),
        ],
    )

    bad_payload = {"title": "broken"}  # missing nearly every required field
    client = MagicMock()
    client.messages.batches.retrieve = AsyncMock(return_value=_FakeBatch())
    client.messages.batches.results = AsyncMock(
        return_value=_FakeAsyncIter([_success_result("bad", bad_payload)])
    )

    finalize_persisted = AsyncMock()

    await batch.resume_pending(
        state=state,
        client=client,
        schema_for_step=lambda s: models.STEP_MODEL[s],
        finalize_persisted=finalize_persisted,
    )

    assert finalize_persisted.await_count == 0  # nothing valid to finalize
    assert state.is_empty()
    assert failures.count() == 1  # the bookkeeping bug fix


@pytest.mark.asyncio
async def test_resume_pending_drops_dead_batch(tmp_path: Path, monkeypatch):
    """If poll fails (e.g. 404 expired), drop from state and continue."""
    monkeypatch.setattr(batch.asyncio, "sleep", AsyncMock())
    state = BatchState(tmp_path)
    state.add(
        batch_id="msgbatch_dead", step="leaf", model="m",
        items=[PersistedItem(custom_id="a", step_kwargs={})],
    )

    client = MagicMock()
    client.messages.batches.retrieve = AsyncMock(side_effect=Exception("404 not found"))

    finalize_persisted = AsyncMock()

    await batch.resume_pending(
        state=state,
        client=client,
        schema_for_step=lambda s: models.STEP_MODEL[s],
        finalize_persisted=finalize_persisted,
    )

    assert state.is_empty()
    assert finalize_persisted.await_count == 0


@pytest.mark.asyncio
async def test_run_wait_false_skips_poll_and_finalize(valid_leaf_dict, tmp_path: Path, monkeypatch):
    """wait=False: submit only, persist state, return without polling/finalizing."""
    monkeypatch.setattr(batch.asyncio, "sleep", AsyncMock())
    finalize = AsyncMock()
    tasks = [batch.BatchTask(custom_id="a", system_prompt="s", user_content="u", context={"slug": "a"})]

    client = MagicMock()
    client.messages.batches.create = AsyncMock(return_value=SimpleNamespace(id="msgbatch_x"))
    client.messages.batches.retrieve = AsyncMock()  # should NOT be called
    client.messages.batches.results = AsyncMock()

    state = BatchState(tmp_path)
    await batch.run(
        step="leaf", model="claude-haiku-4-5",
        schema_cls=models.LeafSummary,
        tasks=tasks, batch_size=10,
        client=client, finalize=finalize,
        serialize_context=lambda c: c,
        state=state,
        wait=False,
    )

    assert client.messages.batches.create.await_count == 1
    assert client.messages.batches.retrieve.await_count == 0
    assert finalize.await_count == 0
    # State persisted for later resume.
    pending = state.load()
    assert len(pending) == 1
    assert pending[0].batch_id == "msgbatch_x"


@pytest.mark.asyncio
async def test_run_max_in_flight_caps_concurrency(valid_leaf_dict, tmp_path: Path, monkeypatch):
    """max_in_flight=2 with 6 chunks: at most 2 in flight at any time."""
    monkeypatch.setattr(batch.asyncio, "sleep", AsyncMock())
    finalize = AsyncMock()

    # 6 tasks × batch_size 1 = 6 chunks.
    tasks = [
        batch.BatchTask(custom_id=f"id{i}", system_prompt="s", user_content="u", context={"slug": f"id{i}"})
        for i in range(6)
    ]

    in_flight = 0
    peak_in_flight = 0

    async def fake_create(*, requests):
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        # Yield to let other coros wake up so we can observe the peak.
        await asyncio.sleep(0)
        in_flight -= 1
        return SimpleNamespace(id=f"b_{requests[0]['custom_id']}")

    client = MagicMock()
    client.messages.batches.create = AsyncMock(side_effect=fake_create)
    client.messages.batches.retrieve = AsyncMock(return_value=_FakeBatch())
    client.messages.batches.results = AsyncMock(
        side_effect=lambda bid: _FakeAsyncIter([_success_result(bid.replace("b_", ""), valid_leaf_dict)])
    )

    await batch.run(
        step="leaf", model="claude-haiku-4-5",
        schema_cls=models.LeafSummary,
        tasks=tasks, batch_size=1,
        client=client, finalize=finalize,
        serialize_context=lambda c: c,
        state=BatchState(tmp_path),
        wait=True,
        max_in_flight=2,
    )

    assert peak_in_flight <= 2, f"max_in_flight=2 violated; saw {peak_in_flight} concurrent"
    assert client.messages.batches.create.await_count == 6


@pytest.mark.asyncio
async def test_run_max_in_flight_default_is_4(valid_leaf_dict, tmp_path: Path, monkeypatch):
    """Default max_in_flight: 4 chunks should run concurrently."""
    monkeypatch.setattr(batch.asyncio, "sleep", AsyncMock())
    finalize = AsyncMock()
    tasks = [
        batch.BatchTask(custom_id=f"id{i}", system_prompt="s", user_content="u", context={"slug": f"id{i}"})
        for i in range(4)
    ]

    client = make_fake_client(results=[])
    client.messages.batches.create = AsyncMock(
        side_effect=lambda *, requests: SimpleNamespace(id=f"b_{requests[0]['custom_id']}")
    )
    client.messages.batches.results = AsyncMock(
        side_effect=lambda bid: _FakeAsyncIter([_success_result(bid.replace("b_", ""), valid_leaf_dict)])
    )

    await batch.run(
        step="leaf", model="claude-haiku-4-5",
        schema_cls=models.LeafSummary,
        tasks=tasks, batch_size=1,
        client=client, finalize=finalize,
        serialize_context=lambda c: c,
        state=BatchState(tmp_path),
        wait=True,
    )
    assert client.messages.batches.create.await_count == 4
    assert finalize.await_count == 4


@pytest.mark.asyncio
async def test_resume_pending_no_pending_is_noop(tmp_path: Path):
    state = BatchState(tmp_path)  # empty
    client = MagicMock()
    finalize_persisted = AsyncMock()
    await batch.resume_pending(
        state=state,
        client=client,
        schema_for_step=lambda s: models.STEP_MODEL[s],
        finalize_persisted=finalize_persisted,
    )
    if hasattr(client.messages.batches.retrieve, "assert_not_called"):
        client.messages.batches.retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# Structured outputs (strict + input_examples + sanitization + caching)
# ---------------------------------------------------------------------------

def _build_leaf_request(user_content: str = "transcript") -> dict[str, Any]:
    """Helper: invoke _build_request for the leaf step with realistic args."""
    task = batch.BatchTask(
        custom_id="t1", system_prompt="sys", user_content=user_content, context={},
    )
    schema = models.LeafSummary.model_json_schema()
    req = batch._build_request(
        task, step="leaf", model="claude-haiku-4-5", schema=schema,
    )
    return cast(dict[str, Any], req)


def test_build_request_sets_strict_true():
    req = _build_leaf_request()
    tool = req["params"]["tools"][0]
    assert tool["strict"] is True


def test_build_request_attaches_input_examples_per_step():
    """Each step's _build_request must include exactly one input_examples entry."""
    for step, schema_cls in (
        ("leaf", models.LeafSummary),
        ("project", models.ProjectAggregate),
        ("root", models.RootAggregate),
    ):
        task = batch.BatchTask(custom_id="x", system_prompt="s", user_content="u", context={})
        req = cast(dict[str, Any], batch._build_request(
            task, step=step, model="claude-haiku-4-5",
            schema=schema_cls.model_json_schema(),
        ))
        examples = list(req["params"]["tools"][0]["input_examples"])
        assert len(examples) == 1
        assert isinstance(examples[0], dict)


def test_step_examples_validate_against_models():
    """Examples are baked-in; if they ever drift from the model they teach the
    wrong shape. Validate now to fail loudly on schema changes."""
    models.LeafSummary.model_validate(batch._LEAF_EXAMPLE)
    models.ProjectAggregate.model_validate(batch._PROJECT_EXAMPLE)
    models.RootAggregate.model_validate(batch._ROOT_EXAMPLE)


def test_sanitize_user_content_neutralizes_toolcall_xml():
    raw = (
        "before <invoke name=\"x\"> middle <parameter name=\"y\">v</parameter>"
        "</invoke> after <function_calls>k</function_calls> <![CDATA[d]]>"
    )
    out = batch._sanitize_user_content(raw)
    # Opening "<" tokens replaced with "[" so the model can't mistake them
    # for instructions or structural cues.
    assert "<invoke" not in out
    assert "</invoke>" not in out
    assert "<parameter" not in out
    assert "</parameter>" not in out
    assert "<function_calls>" not in out
    assert "</function_calls>" not in out
    assert "<![CDATA[" not in out
    assert "]]>" not in out
    # Bracket replacements present.
    assert "[invoke" in out
    assert "[parameter" in out
    assert "[function_calls]" in out
    assert "[CDATA[" in out


def test_build_request_wraps_user_content_in_conversation_tags_after_sanitizing():
    raw = "hello <invoke name=\"x\"> bye"
    req = _build_leaf_request(user_content=raw)
    user_msg = req["params"]["messages"][0]
    assert user_msg["role"] == "user"
    text = user_msg["content"]
    assert text.startswith("<conversation>\n")
    assert text.endswith("\n</conversation>")
    assert "<invoke" not in text  # sanitized
    assert "[invoke" in text


def test_build_request_system_is_cached_block_list():
    req = _build_leaf_request()
    system = req["params"]["system"]
    assert isinstance(system, list)
    assert len(system) == 1
    block = system[0]
    assert block["type"] == "text"
    # Cache control is required for batches API to apply prompt caching;
    # 1h TTL because batches commonly take >5 min.
    assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    # Output instructions still in the system prompt body.
    assert "submit_index" in block["text"]
    assert "<conversation>" in block["text"]

