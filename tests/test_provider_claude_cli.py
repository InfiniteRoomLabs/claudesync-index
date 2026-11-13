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
# Hardening: untrusted transcript text never meets tools (security fix)
# ---------------------------------------------------------------------------

class _Recorder:
    """Stands in for asyncio.create_subprocess_exec. Records the argv, the
    keyword arguments and the stdin payload, and snapshots the contents of
    the subprocess working directory at call time (it is gone afterwards)."""

    def __init__(self, stdout: bytes, *, returncode: int = 0, stderr: bytes = b"", raises: bool = False):
        self.stdout, self.returncode, self.stderr, self.raises = stdout, returncode, stderr, raises
        self.argv: list[str] = []
        self.kwargs: dict = {}
        self.cwd_listing: dict[str, bytes] | None = None
        self.input = ""

    async def __call__(self, *argv, **kwargs):
        self.argv = list(argv)
        self.kwargs = kwargs
        cwd = kwargs.get("cwd")
        if cwd is not None:
            self.cwd_listing = {p.name: p.read_bytes() for p in Path(cwd).iterdir()}
        rec = self

        class _Proc(_FakeProc):
            async def communicate(self, input=None):  # noqa: A002 - mirrors stdlib signature
                rec.input = (input or b"").decode("utf-8")
                if rec.raises:
                    raise RuntimeError("transport broke mid-call")
                return await super().communicate(input)

        return _Proc(self.stdout, self.stderr, self.returncode)

    @property
    def cwd(self) -> Path:
        assert self.kwargs.get("cwd") is not None, "subprocess must run with an explicit cwd"
        return Path(self.kwargs["cwd"])


def _patch_recorder(rec: _Recorder):
    return patch("reindex.providers.claude_cli.asyncio.create_subprocess_exec", new=rec)


def _leaf_schema_str() -> str:
    return json.dumps(models.LeafSummary.model_json_schema())


def _duet_leaf(tmp_export: Path) -> Path:
    """A Duet day directory with the siblings a summarizer must never see."""
    leaf = tmp_export / "duet" / "conversations" / "2026-11-01-general"
    leaf.mkdir(parents=True)
    (leaf / "conversation.md").write_text("## Human\n_2026-11-01_\n\nquoted email text\n", encoding="utf-8")
    for sibling in ("messages.json", "README.md", "INDEX.md", ".claudesync-state.json"):
        (leaf / sibling).write_text("private sibling", encoding="utf-8")
    (tmp_export / "duet" / "duet.json").write_text("{}", encoding="utf-8")
    return leaf


@pytest.mark.asyncio
async def test_inline_mode_runs_restricted_with_no_tools(valid_leaf_dict, tmp_export: Path):
    leaf = _duet_leaf(tmp_export)
    rec = _Recorder(_envelope(valid_leaf_dict).encode())
    with _patch_recorder(rec):
        await _invoke(
            step="leaf", slug="x", model_name="claude-haiku-4-5",
            system_prompt="sys", user_content="inline transcript", work_dir=leaf,
        )
    assert rec.argv == [
        "claude", "-p",
        "--model", "claude-haiku-4-5",
        "--restricted", "--strict-mcp-config", "--no-session-persistence",
        "--tools","",
        "--output-format", "json",
        "--json-schema", _leaf_schema_str(),
    ]
    assert "--disallowed-tools" not in rec.argv
    # The session starts in a fresh, empty directory: never the export root or the leaf.
    assert rec.cwd_listing == {}
    assert not rec.cwd.is_relative_to(tmp_export)
    assert not rec.cwd.exists()  # removed after success


@pytest.mark.asyncio
async def test_file_mode_exposes_only_a_copy_of_conversation_md(valid_leaf_dict, tmp_export: Path):
    leaf = _duet_leaf(tmp_export)
    conv = leaf / "conversation.md"
    rec = _Recorder(_envelope(valid_leaf_dict).encode())
    with _patch_recorder(rec):
        await _invoke(
            step="leaf", slug="x", model_name="claude-haiku-4-5",
            system_prompt=f"Call Read once on `{conv}`. Summarize `{conv}`.",
            user_content=f"The transcript for this conversation is at `{conv}`.",
            work_dir=leaf, allow_filesystem=True,
        )
    sandbox = rec.cwd
    assert rec.argv == [
        "claude", "-p",
        "--model", "claude-haiku-4-5",
        "--restricted", "--strict-mcp-config", "--no-session-persistence",
        "--tools","Read",
        "--add-dir", str(sandbox),
        "--output-format", "json",
        "--json-schema", _leaf_schema_str(),
    ]
    # Only a copy of conversation.md is reachable; its siblings and duet.json are not.
    assert rec.cwd_listing == {"conversation.md": conv.read_bytes()}
    assert not sandbox.is_relative_to(tmp_export)
    # Every mention of the original path now names the copy.
    assert rec.input.count(str(sandbox / "conversation.md")) == 3
    assert str(leaf) not in rec.input
    assert not sandbox.exists()  # removed after success


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_filesystem", [False, True])
@pytest.mark.parametrize("how", ["exit", "exception"])
async def test_sandbox_removed_after_failure(tmp_export: Path, allow_filesystem: bool, how: str):
    leaf = _duet_leaf(tmp_export)
    rec = _Recorder(b"", returncode=2, stderr=b"boom") if how == "exit" else _Recorder(b"", raises=True)
    with _patch_recorder(rec), pytest.raises((ProviderFailure, RuntimeError)):
        await _invoke(
            step="leaf", slug="x", model_name="m",
            system_prompt="s", user_content="u", work_dir=leaf, allow_filesystem=allow_filesystem,
        )
    assert not rec.cwd.exists()


@pytest.mark.asyncio
async def test_unknown_option_asks_for_a_newer_claude_code(tmp_path: Path):
    """An older claude that lacks the hardening flags fails closed: no fallback
    to the unhardened invocation, no escalation retry, a message that says what to do."""
    rec = _Recorder(b"", returncode=1, stderr=b"error: unknown option '--restricted'\n")
    with _patch_recorder(rec), pytest.raises(ProviderFailure) as exc_info:
        await _invoke(
            step="leaf", slug="x", model_name="m",
            system_prompt="s", user_content="u", work_dir=tmp_path,
        )
    err = exc_info.value
    assert err.kind == "process_exit"
    assert err.retryable is False
    assert "newer Claude Code" in str(err)
    for flag in ("--restricted", "--strict-mcp-config", "--tools"):
        assert flag in str(err)
    assert "unknown option" in err.stderr


@pytest.mark.asyncio
async def test_list_envelope_uses_the_last_result_event(valid_leaf_dict, tmp_path: Path):
    """Some claude builds print the whole event stream as a JSON array."""
    earlier = dict(valid_leaf_dict, title="earlier result")
    stream = [
        {"type": "system", "subtype": "init"},
        {"type": "result", "result": json.dumps(earlier), "total_cost_usd": 0.01, "num_turns": 1},
        {"type": "assistant", "message": {}},
        {"type": "result", "result": json.dumps(valid_leaf_dict), "total_cost_usd": 0.07,
         "num_turns": 2, "duration_ms": 99},
        {"type": "system", "subtype": "trailer"},
    ]
    with _patch_subproc(json.dumps(stream).encode()):
        r = await _invoke(
            step="leaf", slug="x", model_name="m",
            system_prompt="s", user_content="u", work_dir=tmp_path,
        )
    assert r.payload.title == "test conversation"
    assert r.cost == 0.07
    assert r.turns == 2
    assert r.duration_ms == 99


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [[], [{"type": "assistant"}], ["result"], 42, "a string"])
async def test_envelope_without_a_result_object_is_envelope_parse(body, tmp_path: Path):
    with _patch_subproc(json.dumps(body).encode()), pytest.raises(ProviderFailure) as exc_info:
        await _invoke(
            step="leaf", slug="x", model_name="m",
            system_prompt="s", user_content="u", work_dir=tmp_path,
        )
    assert exc_info.value.kind == "envelope_parse"
    assert exc_info.value.retryable is False


def test_claude_cli_is_tool_safe():
    assert ClaudeCliProvider.tool_safe is True


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
