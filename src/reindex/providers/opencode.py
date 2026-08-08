"""
opencode provider: `opencode run` subprocess (Groq/Gemini/other via opencode).

NOT BatchCapable — opencode has no server-side batch product.
supports_file_transcripts=False — opencode agents with read permissions are
possible but not configured here; all transcripts are inlined.

Schema enforcement is prompt-only: --output-schema (PR anomalyco/opencode#17276)
is unmerged as of 2026-06-11. The prompt embeds the JSON schema and the local
validate_or_coerce pass catches violations. Expect a higher retry rate compared
to providers with server-side schema enforcement until that PR ships.

Event-stream format (verified by live capture 2026-06-11, opencode v1.17.3):
  Line-delimited JSON (NDJSON) — one object per line.
  {"type":"text", "part": {"type":"text", "text":"<fragment>", ...}}
  {"type":"step_finish", "part": {"type":"step-finish", "tokens":{
      "input": N, "output": N, "reasoning": N, "cache": {"write":N,"read":N}},
      "cost": <float USD>, ...}}
  Text may arrive in multiple "text" events; concatenate all fragments.

  _extract_final_text is the function to revisit if the format changes —
  run `opencode run --format json "Say OK"` with the disable env vars
  to capture a fresh sample.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path

from pydantic import ValidationError

from reindex import log
from reindex.batch import sanitize_user_content
from reindex.providers.base import (
    InvokeRequest,
    InvokeResult,
    Provider,
    ProviderFailure,
    validate_or_coerce,
)
from reindex.providers.claude_cli import _close_subprocess, strip_fences

# Cap captured streams at 4KB. Stderr from opencode is config/startup noise;
# stdout on the failure path is the partial event stream (not the whole run).
_STREAM_CAP = 4096

# Failure kinds where a stronger model plausibly helps — both mean the model
# produced something but not structured output. Process-level failures aren't
# retryable because a bigger model won't fix a missing binary or auth error.
_RETRYABLE_KINDS = frozenset({"result_parse", "schema_violation"})


def _extract_final_text(stream_output: str) -> tuple[str, int, int, float]:
    """Parse an opencode --format json NDJSON event stream.

    Returns (final_text, input_tokens, output_tokens, cost_usd).

    The stream is line-delimited JSON. Text arrives in one or more events:
      {"type":"text", "part": {"type":"text", "text": "<fragment>"}}
    Token counts and cost arrive in the step_finish event:
      {"type":"step_finish", "part": {"tokens": {...}, "cost": <float>}}

    Tolerant: lines that are not valid JSON or lack the expected keys are
    skipped rather than fatal. If NO text events parsed, returns the whole
    stdout as the text so the caller's JSON decoder gets a chance on plain
    output (second-chance fallback for format changes or single-line output).
    """
    fragments: list[str] = []
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    any_event_parsed = False

    for raw_line in stream_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")
        part = event.get("part", {})

        if event_type == "text":
            # Text fragment from the assistant message.
            text = part.get("text", "")
            if text:
                fragments.append(text)
            any_event_parsed = True

        elif event_type == "step_finish":
            # Final accounting: token counts + USD cost.
            tokens = part.get("tokens", {})
            # input and output are the "raw" (non-cached) token counts;
            # cache.read/cache.write are tracked separately by opencode.
            # For cost accounting purposes use whatever opencode reports.
            input_tokens = int(tokens.get("input", 0))
            output_tokens = int(tokens.get("output", 0))
            cost_usd = float(part.get("cost", 0.0))
            any_event_parsed = True

    if fragments:
        return "".join(fragments), input_tokens, output_tokens, cost_usd

    if any_event_parsed:
        # Events parsed but no text fragments — this is unusual (e.g. the
        # model replied with only tool calls). Return empty string to let
        # the caller raise result_parse.
        return "", input_tokens, output_tokens, cost_usd

    # No NDJSON events parsed at all: treat the whole output as a bare text
    # response. Handles the hypothetical case where opencode changes its
    # output format or --format json is ignored by a future version.
    return stream_output.strip(), 0, 0, 0.0


def _quick_agent_path() -> Path:
    """Resolve the path the optional `--agent quick` flag depends on.

    Honors $XDG_CONFIG_HOME (falling back to ~/.config) instead of
    hardcoding the config location, so this stays correct on any machine
    opencode's config directory conventions apply to.
    """
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
    return base / "opencode" / "agent" / "quick.md"


class OpencodeProvider(Provider):
    """opencode run subprocess provider.

    Delivers the combined prompt via stdin (opencode reads stdin when no
    message args are given). When a `quick` agent file is configured (see
    `_quick_agent_path`), passes --agent quick to strip opencode's built-in
    tool permissions and cut prompt overhead; when it's absent, the flag is
    omitted and a one-line warning is logged instead of failing. Env vars
    suppress the CLAUDE.md + external-skills bloat that would otherwise add
    tokens to every call regardless of the agent flag.
    """

    name = "opencode"
    supports_file_transcripts = False  # inline only; --add-dir not configured

    async def preflight(self) -> None:
        # The shim is under ~/.local/share/mise/shims/; PATH may not include
        # it in all env configs. Surface a clear error rather than a confusing
        # "opencode: command not found" from inside asyncio.create_subprocess_exec.
        if shutil.which("opencode") is None:
            raise ProviderFailure(
                "`opencode` binary not on PATH — run `mise up opencode` to install it",
                kind="process_exit",
                provider=self.name,
            )

    def _fail(self, message: str, *, kind: str, **kw) -> ProviderFailure:
        return ProviderFailure(
            message,
            kind=kind,
            provider=self.name,
            retryable=kind in _RETRYABLE_KINDS,
            **kw,
        )

    @staticmethod
    def _build_argv(model: str = "") -> list[str]:
        """Build the `opencode run` argv, degrading gracefully without a
        configured `quick` agent.

        --agent quick is included only when `_quick_agent_path()` exists.
        A fresh opencode install (or a CI environment) has no such file,
        and that's a purely optional cost optimization, not a hard
        requirement — the subprocess should still run against opencode's
        default agent rather than fail outright.
        """
        argv = ["opencode", "run", "--format", "json"]
        if _quick_agent_path().is_file():
            argv.extend(["--agent", "quick"])
        else:
            log.get("provider.opencode").warning(
                "opencode_agent_missing",
                hint="create a no-tools agent named 'quick' to cut prompt overhead",
            )
        if model:
            argv.extend(["--model", model])
        return argv

    async def invoke(self, req: InvokeRequest) -> InvokeResult:
        """opencode run subprocess with prompt-embedded JSON schema.

        The combined prompt (system + framed transcript + OUTPUT instruction
        with schema) is fed via stdin. The event stream is parsed by
        _extract_final_text, then the assistant text goes through strip_fences
        + JSON decode + validate_or_coerce — the same pipeline claude-cli uses
        after unwrapping its envelope.

        req.allow_filesystem and req.work_dir are ignored: this provider
        always inlines transcripts (supports_file_transcripts=False).
        """
        blog = log.get("backend").bind(step=req.step, slug=req.slug)
        t0 = time.monotonic()

        schema = req.schema_cls.model_json_schema()
        schema_str = json.dumps(schema)

        # Wrap transcript in <conversation> tags + sanitize XML markers so
        # any literal <invoke>/<parameter> spans in prior tool-use history
        # can't be misread as a fresh tool call by the summarising model.
        framed_transcript = (
            "<conversation>\n"
            + sanitize_user_content(req.user_content)
            + "\n</conversation>"
        )
        combined = (
            f"{req.system_prompt}\n\n{framed_transcript}\n\n"
            "OUTPUT: Respond with a single JSON object only, no prose, no markdown fences. "
            f"The JSON must conform to this schema:\n{schema_str}"
        )

        # --agent quick (when configured — see _build_argv) points at an
        # agent file that denies ALL tools (read/edit/bash/glob/grep/…).
        # Without it, opencode's default agents have read/write/bash
        # permissions and a larger harness floor is paid even when none of
        # those tools are exercised.
        #
        # Env vars suppress two prompt-bloat sources independent of the
        # agent flag:
        #   OPENCODE_DISABLE_CLAUDE_CODE=1    — don't load ~/.claude/CLAUDE.md
        #   OPENCODE_DISABLE_EXTERNAL_SKILLS=1 — skip ~/.agents/skills (firecrawl etc.)
        # Together with --agent quick these cut the request floor from ~32k to ~159
        # input tokens (measured live against opencode v1.17.3 + groq/gpt-oss-120b).
        cli_args = self._build_argv(req.model)
        subprocess_env = {
            **os.environ,
            "OPENCODE_DISABLE_CLAUDE_CODE": "1",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        }

        proc = await asyncio.create_subprocess_exec(
            *cli_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=subprocess_env,
        )
        try:
            stdout, stderr = await proc.communicate(input=combined.encode("utf-8"))
        finally:
            await _close_subprocess(proc)

        duration_ms = int((time.monotonic() - t0) * 1000)
        stderr_text = stderr.decode("utf-8", errors="replace")[:_STREAM_CAP]
        stdout_text = stdout.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            blog.error(
                "opencode_failed",
                exit=proc.returncode,
                stderr=stderr_text,
                stdout_preview=stdout_text[:500],
            )
            raise self._fail(
                f"opencode exit {proc.returncode}",
                kind="process_exit",
                exit_code=proc.returncode,
                stderr=stderr_text,
                stdout=stdout_text[:_STREAM_CAP],
            )

        final_text, in_tok, out_tok, event_cost = _extract_final_text(stdout_text)
        raw = strip_fences(final_text)

        try:
            tool_input = json.loads(raw)
        except json.JSONDecodeError as e:
            blog.error(
                "opencode_non_json",
                raw=raw[:300],
                stderr=stderr_text[:500],
                stdout_preview=stdout_text[:500],
            )
            raise self._fail(
                f"opencode final text is not JSON: {e}",
                kind="result_parse",
                exit_code=proc.returncode,
                stderr=stderr_text,
                stdout=raw[:_STREAM_CAP],
            ) from e

        try:
            payload = validate_or_coerce(
                tool_input, req.schema_cls, log_bind=blog, custom_id=req.slug
            )
        except ValidationError as e:
            blog.error("opencode_schema_violation", errors=str(e)[:300])
            raise self._fail(
                f"schema violation: {e}",
                kind="schema_violation",
                exit_code=proc.returncode,
                stderr=stderr_text,
                stdout=raw[:_STREAM_CAP],
            ) from e

        # Use the cost reported by opencode's step_finish event when available
        # (it incorporates the provider's cache discounts for repeated runs).
        # Fall back to the token-table estimate if the event stream had no cost.
        cost = event_cost if event_cost > 0 else self.config.compute_cost(
            req.model, in_tok, out_tok
        )

        return InvokeResult(
            payload=payload,
            cost=cost,
            duration_ms=duration_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )
