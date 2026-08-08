"""opencode provider: opencode run subprocess + NDJSON event stream parsing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from reindex import config, models
from reindex.providers.base import InvokeRequest, ProviderFailure
from reindex.providers.opencode import OpencodeProvider, _extract_final_text, _quick_agent_path


def _provider() -> OpencodeProvider:
    return OpencodeProvider(
        config.load(Path("/nonexistent"), provider_name=config.ProviderName.OPENCODE)
    )


async def _invoke(
    *, step, slug, model_name, system_prompt, user_content, work_dir=None
):
    return await _provider().invoke(
        InvokeRequest(
            step=step,
            slug=slug,
            model=model_name,
            system_prompt=system_prompt,
            user_content=user_content,
            schema_cls=models.STEP_MODEL[step],
            work_dir=work_dir,
        )
    )


def _event_stream(text: str, *, input_tokens: int = 100, output_tokens: int = 50, cost: float = 0.01) -> bytes:
    """Build a minimal NDJSON event stream matching opencode v1.17.3 format."""
    events = [
        json.dumps({
            "type": "step_start",
            "timestamp": 123456,
            "part": {"type": "step-start", "messageID": "msg_test"},
        }),
        json.dumps({
            "type": "text",
            "timestamp": 123456,
            "part": {"type": "text", "text": text},
        }),
        json.dumps({
            "type": "step_finish",
            "timestamp": 123457,
            "part": {
                "type": "step-finish",
                "reason": "stop",
                "tokens": {
                    "total": input_tokens + output_tokens,
                    "input": input_tokens,
                    "output": output_tokens,
                    "reasoning": 0,
                    "cache": {"write": 0, "read": 0},
                },
                "cost": cost,
            },
        }),
    ]
    return "\n".join(events).encode("utf-8")


class _FakeProc:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self, input=None):  # noqa: A002
        _ = input
        return self._stdout, self._stderr


def _patch_subproc(stdout: bytes, *, returncode: int = 0, stderr: bytes = b""):
    fake = _FakeProc(stdout, stderr, returncode)
    return patch(
        "reindex.providers.opencode.asyncio.create_subprocess_exec",
        AsyncMock(return_value=fake),
    )


# ---------------------------------------------------------------------------
# Happy path: NDJSON event stream
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invoke_success_event_stream(valid_leaf_dict, tmp_path: Path):
    """Full happy path: NDJSON stream → stripped fences → validate → InvokeResult."""
    stream = _event_stream(json.dumps(valid_leaf_dict), input_tokens=159, output_tokens=42, cost=0.00012)
    with _patch_subproc(stream):
        r = await _invoke(
            step="leaf", slug="x", model_name="google/gemini-2.5-flash",
            system_prompt="sys", user_content="user", work_dir=tmp_path,
        )
    assert isinstance(r.payload, models.LeafSummary)
    assert r.payload.title == "test conversation"
    # Cost comes from the step_finish event, not the token table.
    assert r.cost == pytest.approx(0.00012)
    assert r.input_tokens == 159
    assert r.output_tokens == 42


@pytest.mark.asyncio
async def test_invoke_success_multi_fragment_stream(valid_leaf_dict):
    """Multiple text events in the stream — fragments are concatenated."""
    json_str = json.dumps(valid_leaf_dict)
    # Split the JSON across three text events.
    mid = len(json_str) // 2
    events = [
        json.dumps({"type": "step_start", "part": {"type": "step-start"}}),
        json.dumps({"type": "text", "part": {"type": "text", "text": json_str[:mid]}}),
        json.dumps({"type": "text", "part": {"type": "text", "text": json_str[mid:]}}),
        json.dumps({
            "type": "step_finish",
            "part": {
                "type": "step-finish", "reason": "stop",
                "tokens": {"input": 50, "output": 80, "cache": {"write": 0, "read": 0}},
                "cost": 0.005,
            },
        }),
    ]
    stdout = "\n".join(events).encode("utf-8")
    with _patch_subproc(stdout):
        r = await _invoke(
            step="leaf", slug="x", model_name="google/gemini-2.5-flash",
            system_prompt="s", user_content="u",
        )
    assert isinstance(r.payload, models.LeafSummary)
    assert r.payload.title == "test conversation"


@pytest.mark.asyncio
async def test_invoke_success_plain_json_fallback(valid_leaf_dict):
    """Fallback path: stdout is bare JSON text with no NDJSON events."""
    plain = json.dumps(valid_leaf_dict).encode("utf-8")
    with _patch_subproc(plain):
        r = await _invoke(
            step="leaf", slug="x", model_name="google/gemini-2.5-flash",
            system_prompt="s", user_content="u",
        )
    assert isinstance(r.payload, models.LeafSummary)
    # No token data from the stream; cost falls back to compute_cost (0 for unknowns).
    assert r.input_tokens == 0
    assert r.output_tokens == 0


@pytest.mark.asyncio
async def test_fenced_json_stripped(valid_leaf_dict):
    """```json ... ``` fences in the assistant text are stripped before parsing."""
    fenced = "```json\n" + json.dumps(valid_leaf_dict) + "\n```"
    stream = _event_stream(fenced)
    with _patch_subproc(stream):
        r = await _invoke(
            step="leaf", slug="x", model_name="google/gemini-2.5-flash",
            system_prompt="s", user_content="u",
        )
    assert isinstance(r.payload, models.LeafSummary)
    assert r.payload.title == "test conversation"


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nonzero_exit_raises_process_exit():
    """Nonzero exit maps to kind=process_exit (not retryable)."""
    with _patch_subproc(b"", returncode=1, stderr=b"auth error"):
        with pytest.raises(ProviderFailure) as exc_info:
            await _invoke(
                step="leaf", slug="x", model_name="m",
                system_prompt="s", user_content="u",
            )
    assert exc_info.value.kind == "process_exit"
    assert exc_info.value.retryable is False
    assert exc_info.value.exit_code == 1
    assert "auth error" in exc_info.value.stderr


@pytest.mark.asyncio
async def test_prose_final_text_raises_result_parse():
    """Model returns prose (not JSON) → kind=result_parse, retryable=True."""
    stream = _event_stream("Sorry, I cannot help with that.")
    with _patch_subproc(stream):
        with pytest.raises(ProviderFailure) as exc_info:
            await _invoke(
                step="leaf", slug="x", model_name="m",
                system_prompt="s", user_content="u",
            )
    assert exc_info.value.kind == "result_parse"
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_schema_garbage_raises_schema_violation(valid_leaf_dict):
    """Schema-incompatible JSON (missing required field) → kind=schema_violation.

    Must be a field coercion can't repair (title is required and can't be
    synthesized) so validate_or_coerce actually raises ValidationError.
    """
    del valid_leaf_dict["title"]
    stream = _event_stream(json.dumps(valid_leaf_dict))
    with _patch_subproc(stream):
        with pytest.raises(ProviderFailure) as exc_info:
            await _invoke(
                step="leaf", slug="x", model_name="m",
                system_prompt="s", user_content="u",
            )
    assert exc_info.value.kind == "schema_violation"
    assert exc_info.value.retryable is True
    assert isinstance(exc_info.value.__cause__, ValidationError)


@pytest.mark.asyncio
async def test_coercible_shape_error_succeeds(valid_leaf_dict):
    """string-where-array (topics) is repaired by validate_or_coerce — no failure."""
    valid_leaf_dict["topics"] = "single-topic"
    stream = _event_stream(json.dumps(valid_leaf_dict))
    with _patch_subproc(stream):
        r = await _invoke(
            step="leaf", slug="x", model_name="m",
            system_prompt="s", user_content="u",
        )
    assert r.payload.topics == ["single-topic"]


# ---------------------------------------------------------------------------
# Subprocess argv and env assertions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subprocess_argv_and_env(
    valid_leaf_dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """opencode run --format json --agent quick --model <model> must appear in argv
    when the quick agent file is configured. Both OPENCODE_DISABLE_* env vars must
    be set in the subprocess environment.
    """
    agent_dir = tmp_path / "opencode" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "quick.md").write_text("---\ntools: {}\n---\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    stream = _event_stream(json.dumps(valid_leaf_dict))
    fake = _FakeProc(stream)
    mock_exec = AsyncMock(return_value=fake)

    with patch("reindex.providers.opencode.asyncio.create_subprocess_exec", mock_exec):
        await _invoke(
            step="leaf", slug="slug-1", model_name="google/gemini-2.5-flash",
            system_prompt="sys", user_content="user",
        )

    call_kwargs = mock_exec.call_args
    argv = call_kwargs.args  # positional *args are the argv tokens
    env = call_kwargs.kwargs.get("env", {})

    assert argv[0] == "opencode"
    assert "--format" in argv
    assert "json" in argv
    assert "--agent" in argv
    assert "quick" in argv
    assert "--model" in argv
    assert "google/gemini-2.5-flash" in argv

    # Both disable vars must be present to suppress bloat.
    assert env.get("OPENCODE_DISABLE_CLAUDE_CODE") == "1"
    assert env.get("OPENCODE_DISABLE_EXTERNAL_SKILLS") == "1"

    # Subprocess env should inherit the caller's environment, not start fresh.
    assert "PATH" in env


def test_build_argv_includes_agent_flag_when_agent_file_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """--agent quick is included when _quick_agent_path() resolves to a real file."""
    agent_dir = tmp_path / "opencode" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "quick.md").write_text("---\ntools: {}\n---\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    argv = OpencodeProvider._build_argv("google/gemini-2.5-flash")

    assert "--agent" in argv
    assert "quick" in argv


def test_build_argv_omits_agent_flag_when_agent_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """No ~/.config/opencode/agent/quick.md -> flag is omitted and a warning is logged."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert not _quick_agent_path().is_file()

    mock_logger = MagicMock()
    with patch("reindex.providers.opencode.log.get", return_value=mock_logger):
        argv = OpencodeProvider._build_argv()

    assert "--agent" not in argv
    assert "quick" not in argv
    mock_logger.warning.assert_called_once()
    assert mock_logger.warning.call_args.args[0] == "opencode_agent_missing"


# ---------------------------------------------------------------------------
# _extract_final_text unit tests
# ---------------------------------------------------------------------------

def test_extract_text_from_ndjson_events():
    """Standard event stream: text fragments concatenated, tokens from step_finish."""
    stream = "\n".join([
        json.dumps({"type": "step_start", "part": {}}),
        json.dumps({"type": "text", "part": {"text": "hello "}}),
        json.dumps({"type": "text", "part": {"text": "world"}}),
        json.dumps({
            "type": "step_finish",
            "part": {
                "tokens": {"input": 10, "output": 5, "cache": {}},
                "cost": 0.001,
            },
        }),
    ])
    text, in_tok, out_tok, cost = _extract_final_text(stream)
    assert text == "hello world"
    assert in_tok == 10
    assert out_tok == 5
    assert cost == pytest.approx(0.001)


def test_extract_fallback_on_plain_text():
    """Non-NDJSON stdout falls back to the whole string with zero accounting."""
    text, in_tok, out_tok, cost = _extract_final_text('{"foo": "bar"}')
    assert text == '{"foo": "bar"}'
    assert in_tok == 0
    assert out_tok == 0
    assert cost == 0.0


def test_extract_skips_malformed_lines():
    """Non-JSON lines in the stream are silently skipped."""
    stream = "\n".join([
        "not json at all",
        json.dumps({"type": "text", "part": {"text": "OK"}}),
        json.dumps({
            "type": "step_finish",
            "part": {"tokens": {"input": 1, "output": 1, "cache": {}}, "cost": 0.0},
        }),
    ])
    text, _, _, _ = _extract_final_text(stream)
    assert text == "OK"


def test_extract_empty_stream():
    """Empty stdout returns empty text with zero accounting."""
    text, in_tok, out_tok, cost = _extract_final_text("")
    assert text == ""
    assert in_tok == 0
    assert cost == 0.0


# ---------------------------------------------------------------------------
# Provider metadata
# ---------------------------------------------------------------------------

def test_provider_name():
    assert OpencodeProvider.name == "opencode"


def test_provider_not_batch_capable():
    from reindex.providers.base import BatchCapable
    assert not isinstance(_provider(), BatchCapable)


def test_provider_not_file_transcripts():
    assert OpencodeProvider.supports_file_transcripts is False
