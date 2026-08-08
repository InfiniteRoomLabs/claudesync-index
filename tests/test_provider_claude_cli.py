"""claude-cli provider: claude -p subprocess + JSON envelope parsing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from reindex import config, models
from reindex.providers.base import InvokeRequest, ProviderFailure
from reindex.providers.claude_cli import ClaudeCliProvider, strip_fences


def _provider() -> ClaudeCliProvider:
    return ClaudeCliProvider(config.load(Path("/nonexistent"), provider_name=config.ProviderName.CLAUDE_CLI))


async def _invoke(*, step, slug, model_name, system_prompt, user_content, work_dir, allow_filesystem=False):
    return await _provider().invoke(InvokeRequest(
        step=step, slug=slug, model=model_name,
        system_prompt=system_prompt, user_content=user_content,
        schema_cls=models.STEP_MODEL[step],
        work_dir=work_dir, allow_filesystem=allow_filesystem,
    ))


def _envelope(result_json: dict, *, cost: float = 0.05, turns: int = 1, duration_ms: int = 1234) -> str:
    return json.dumps({
        "total_cost_usd": cost,
        "num_turns": turns,
        "duration_ms": duration_ms,
        "result": json.dumps(result_json),
    })


class _FakeProc:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self, input=None):  # noqa: A002 - mirrors stdlib signature
        _ = input
        return self._stdout, self._stderr


def _patch_subproc(stdout: bytes, *, returncode: int = 0, stderr: bytes = b""):
    fake = _FakeProc(stdout, stderr, returncode)
    return patch(
        "reindex.providers.claude_cli.asyncio.create_subprocess_exec",
        AsyncMock(return_value=fake),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subscription_invoke_success(valid_leaf_dict, tmp_path: Path):
    stdout = _envelope(valid_leaf_dict).encode()
    with _patch_subproc(stdout):
        r = await _invoke(
            step="leaf", slug="x", model_name="claude-haiku-4-5",
            system_prompt="sys", user_content="user",
            work_dir=tmp_path,
        )
    assert isinstance(r.payload, models.LeafSummary)
    assert r.payload.title == "test conversation"
    assert r.cost == 0.05
    assert r.turns == 1


@pytest.mark.asyncio
async def test_strip_fences_around_result(valid_leaf_dict, tmp_path: Path):
    """claude sometimes wraps JSON in ```json ... ``` even when told not to."""
    fenced = "```json\n" + json.dumps(valid_leaf_dict) + "\n```"
    envelope = json.dumps({
        "total_cost_usd": 0.01, "num_turns": 1, "duration_ms": 100,
        "result": fenced,
    })
    with _patch_subproc(envelope.encode()):
        r = await _invoke(
            step="leaf", slug="x", model_name="claude-haiku-4-5",
            system_prompt="s", user_content="u", work_dir=tmp_path,
        )
    assert isinstance(r.payload, models.LeafSummary)
    assert r.payload.title == "test conversation"


@pytest.mark.asyncio
async def test_strip_fences_no_language_hint(valid_leaf_dict, tmp_path: Path):
    fenced = "```\n" + json.dumps(valid_leaf_dict) + "\n```"
    envelope = json.dumps({
        "total_cost_usd": 0.01, "num_turns": 1, "duration_ms": 100,
        "result": fenced,
    })
    with _patch_subproc(envelope.encode()):
        r = await _invoke(
            step="leaf", slug="x", model_name="claude-haiku-4-5",
            system_prompt="s", user_content="u", work_dir=tmp_path,
        )
    assert isinstance(r.payload, models.LeafSummary)
    assert r.payload.title == "test conversation"


# ---------------------------------------------------------------------------
# Negative paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_claude_nonzero_exit_raises(tmp_path: Path):
    with _patch_subproc(b"", returncode=2, stderr=b"unauthorized"):
        with pytest.raises(ProviderFailure) as exc_info:
            await _invoke(
                step="leaf", slug="x", model_name="m",
                system_prompt="s", user_content="u", work_dir=tmp_path,
            )
    # Exit code + stderr should be on the structured exception so the
    # failure log captures them instead of just "claude -p exit 2".
    assert exc_info.value.kind == "process_exit"
    assert exc_info.value.exit_code == 2
    assert "unauthorized" in exc_info.value.stderr


@pytest.mark.asyncio
async def test_failure_context_serializes_for_failure_log(tmp_path: Path):
    """to_context() shape is what failures.record stores; pin it so we
    don't accidentally regress the on-disk failure-log schema."""
    with _patch_subproc(b"", returncode=137, stderr=b"oom-killer struck"):
        with pytest.raises(ProviderFailure) as exc_info:
            await _invoke(
                step="leaf", slug="x", model_name="m",
                system_prompt="s", user_content="u", work_dir=tmp_path,
            )
    ctx = exc_info.value.to_context()
    assert ctx["kind"] == "process_exit"
    assert ctx["exit_code"] == 137
    assert ctx["stderr"] == "oom-killer struck"
    assert ctx["stdout"] == ""
    assert "claude -p exit 137" in str(ctx["message"])


@pytest.mark.asyncio
async def test_non_json_envelope_raises(tmp_path: Path):
    with _patch_subproc(b"not json"):
        with pytest.raises(ProviderFailure) as exc_info:
            await _invoke(
                step="leaf", slug="x", model_name="m",
                system_prompt="s", user_content="u", work_dir=tmp_path,
            )
    assert exc_info.value.kind == "envelope_parse"
    assert "not json" in exc_info.value.stdout


@pytest.mark.asyncio
async def test_result_field_not_json(tmp_path: Path):
    envelope = json.dumps({
        "total_cost_usd": 0.01, "num_turns": 1, "duration_ms": 100,
        "result": "this is not JSON",
    })
    with _patch_subproc(envelope.encode()):
        with pytest.raises(ProviderFailure) as exc_info:
            await _invoke(
                step="leaf", slug="x", model_name="m",
                system_prompt="s", user_content="u", work_dir=tmp_path,
            )
    assert exc_info.value.kind == "result_parse"


@pytest.mark.asyncio
async def test_schema_violation_propagates(valid_leaf_dict, tmp_path: Path):
    """Schema violations surface as BackendFailure so the worker gets the
    structured context (kind / stderr / stdout) for the failure log. Must
    be a violation coercion can't repair (missing required field) -- the
    provider now runs validate_or_coerce before failing.
    The underlying ValidationError is still chained via __cause__."""
    del valid_leaf_dict["title"]  # required; coercion can't invent it
    envelope = _envelope(valid_leaf_dict).encode()
    with _patch_subproc(envelope):
        with pytest.raises(ProviderFailure) as exc_info:
            await _invoke(
                step="leaf", slug="x", model_name="m",
                system_prompt="s", user_content="u", work_dir=tmp_path,
            )
    assert exc_info.value.kind == "schema_violation"
    assert isinstance(exc_info.value.__cause__, ValidationError)


@pytest.mark.asyncio
async def test_coercible_shape_error_now_succeeds(valid_leaf_dict, tmp_path: Path):
    """string-where-array is a mechanical shape error: the provider's
    validate_or_coerce pass repairs it locally instead of burning an
    escalation retry (pre-provider behavior was schema_violation)."""
    valid_leaf_dict["topics"] = "string-not-array"
    envelope = _envelope(valid_leaf_dict).encode()
    with _patch_subproc(envelope):
        r = await _invoke(
            step="leaf", slug="x", model_name="m",
            system_prompt="s", user_content="u", work_dir=tmp_path,
        )
    assert r.payload.topics == ["string-not-array"]


# ---------------------------------------------------------------------------
# _strip_fences edge cases
# ---------------------------------------------------------------------------

def test_strip_fences_no_fence():
    s = '{"a": 1}'
    assert strip_fences(s) == s


def test_strip_fences_only_opening():
    s = "```json\n{\"a\": 1}"
    out = strip_fences(s)
    assert out == '{"a": 1}'


def test_strip_fences_handles_whitespace():
    s = "  \n```json\n{\"a\": 1}\n```\n  "
    assert strip_fences(s) == '{"a": 1}'
