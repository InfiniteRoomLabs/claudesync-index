"""
claude-cli provider: `claude -p` subprocess (subscription quota).

The only provider with supports_file_transcripts — oversized transcripts
are delivered via the Read tool instead of inlined, from a private copy in
a throwaway directory. Structured output via --json-schema plus a local
validate/coerce pass.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

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

# Cap captured streams at 4KB each. Real claude -p stderr is rarely more
# than a few lines; stdout on the parse-failure path is usually a JSON
# envelope a few KB long. 4KB per stream keeps the failure log readable
# without truncating useful diagnostics.
_STREAM_CAP = 4096

# Failure kinds where a stronger model plausibly fixes the second attempt:
# both mean the model produced *something* but not what we asked for.
# Process-level failures (process_exit, envelope_parse) are NOT retryable:
# process_exit on a too-long prompt is already routed around via file
# mode, and envelope_parse signals a claude -p CLI bug that a bigger
# model won't fix.
_RETRYABLE_KINDS = frozenset({"result_parse", "schema_violation"})

# The flags that keep untrusted transcript text away from tools. A claude
# too old to know them fails with commander's "unknown option" error; that
# is reported as an upgrade request, never retried without them.
_HARDENING_FLAGS = ("--restricted", "--strict-mcp-config", "--tools")
_UNKNOWN_OPTION = re.compile(r"unknown option", re.IGNORECASE)


def _result_envelope(parsed: Any) -> dict | None:
    """The result envelope from `--output-format json` output. Most builds
    print one object; some print the whole event stream as a JSON array, in
    which case the last `"type": "result"` event is the envelope."""
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        results = [e for e in parsed if isinstance(e, dict) and e.get("type") == "result"]
        return results[-1] if results else None
    return None


async def _close_subprocess(proc: asyncio.subprocess.Process) -> None:
    """Reap and explicitly close a Process and its underlying transport.

    asyncio.subprocess.Process never closes its transport on its own;
    communicate() and wait() both leave it for GC. If GC fires after
    asyncio.run() returns, the transport's __del__ calls loop.call_soon()
    on a closed loop and emits 'RuntimeError: Event loop is closed' per
    leaked subprocess.

    Call this in a finally block around every subprocess use so the
    transport is closed while the loop is still alive. Safe to call on
    both success and cancellation paths -- the reap step is a no-op
    once the child has exited.
    """
    # Cancel path: reap the child first
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        with contextlib.suppress(BaseException):
            await proc.wait()
    # Both paths: explicitly close the transport while the loop is alive.
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        with contextlib.suppress(BaseException):
            # noinspection PyUnresolvedReferences
            transport.close()


def strip_fences(s: str) -> str:
    """Remove surrounding markdown ``` fences from a model response."""
    s = s.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl >= 0:
            s = s[nl + 1:]
        if s.endswith("```"):
            s = s[: -len("```")].rstrip()
    return s


class ClaudeCliProvider(Provider):
    name = "claude-cli"
    supports_file_transcripts = True
    tool_safe = True  # --restricted --strict-mcp-config --tools "" (or Read on a private copy)

    async def preflight(self) -> None:
        if shutil.which("claude") is None:
            raise ProviderFailure(
                "`claude` binary not on PATH (https://claude.com/claude-code)",
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

    async def invoke(self, req: InvokeRequest) -> InvokeResult:
        """claude -p subprocess with --json-schema.

        req.allow_filesystem: when True (file-mode leaves, transcripts too
        large to inline), `<req.work_dir>/conversation.md` is copied into a
        throwaway directory and the model may call Read on that copy only.
        Otherwise the model has no tools at all. Either way the subprocess
        runs in a fresh temporary directory, removed afterwards.
        """
        with tempfile.TemporaryDirectory(prefix="csindex-claude-") as sandbox:
            return await self._invoke_in(Path(sandbox), req)

    async def _invoke_in(self, sandbox: Path, req: InvokeRequest) -> InvokeResult:
        blog = log.get("backend").bind(step=req.step, slug=req.slug)
        schema = req.schema_cls.model_json_schema()
        schema_str = json.dumps(schema)
        # Wrap the transcript in <conversation> tags so the model sees it as
        # input DATA, not as a fresh user request. Run the same XML-marker
        # sanitizer the batch path uses so any literal <invoke>/<parameter>
        # spans inside the transcript can't be misread as a tool call.
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

        # The transcript is untrusted text: a quoted email, a pasted web page
        # or a Duet message can carry instructions. It must never meet a tool,
        # an MCP server or the user's permission mode. Verified on 2026-11-13
        # against claude 2.1.325: the old `--disallowed-tools` invocation
        # started with 266 tools, the user's MCP servers and the user's
        # permission mode; the invocation below starts with exactly one tool
        # (StructuredOutput), no MCP servers and the default permission mode.
        #   * --restricted --strict-mcp-config: always. No --mcp-config is
        #     given, so no MCP server (mail, chat, ...) loads.
        #   * --tools "" (inline): no built-in tools; the model can answer only
        #     through --json-schema's structured output.
        #   * --tools Read (file mode, oversized transcripts): the one tool the
        #     file-mode prompt needs. conversation.md is copied into the
        #     sandbox, the sandbox is the only --add-dir, and the prompt is
        #     rewritten to name the copy, so the leaf's siblings
        #     (messages.json, README.md) and the export root stay unreachable.
        #   * cwd=<sandbox>: a fresh temporary directory in both modes, so the
        #     session's working directory never holds export content.
        #   * --no-session-persistence: the session is not saved. Every call
        #     runs from a new directory, so a saved session would leave one
        #     ~/.claude/projects entry (holding the transcript) per summary.
        #     Verified 2026-11-13: the entry count is unchanged after a call,
        #     and in file mode a Read outside the sandbox is denied.
        # Never fall back to a weaker invocation: an older claude that rejects
        # these flags gets an upgrade message (below), not a retry without them.
        cli_args = [
            "claude", "-p",
            "--model", req.model,
            "--restricted", "--strict-mcp-config", "--no-session-persistence",
        ]
        if req.allow_filesystem and req.work_dir is not None:
            original = req.work_dir / "conversation.md"
            copy = sandbox / "conversation.md"
            shutil.copyfile(original, copy)
            combined = combined.replace(str(original), str(copy))
            cli_args.extend(["--tools", "Read", "--add-dir", str(sandbox)])
        else:
            cli_args.extend(["--tools", ""])
        cli_args.extend(["--output-format", "json", "--json-schema", schema_str])

        proc = await asyncio.create_subprocess_exec(
            *cli_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(sandbox),
        )
        try:
            stdout, stderr = await proc.communicate(input=combined.encode("utf-8"))
        finally:
            await _close_subprocess(proc)
        stderr_text = stderr.decode("utf-8", errors="replace")[:_STREAM_CAP]
        stdout_text = stdout.decode("utf-8", errors="replace")[:_STREAM_CAP]

        if proc.returncode != 0:
            blog.error(
                "subscription_claude_failed",
                exit=proc.returncode,
                stderr=stderr_text,
                stdout_preview=stdout_text[:500],
            )
            message = f"claude -p exit {proc.returncode}"
            if _UNKNOWN_OPTION.search(stderr_text):
                message = (
                    f"{message}: this Claude Code does not support {', '.join(_HARDENING_FLAGS)}, "
                    "which csindex needs to run the summarizer without tools. "
                    "Install a newer Claude Code (`claude update`) and rerun."
                )
            raise self._fail(
                message,
                kind="process_exit",
                exit_code=proc.returncode,
                stderr=stderr_text,
                stdout=stdout_text,
            )

        try:
            parsed = json.loads(stdout.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            blog.error("subscription_envelope_parse", error=str(e)[:300], stdout_preview=stdout_text[:500])
            raise self._fail(
                f"claude -p returned non-JSON envelope: {e}",
                kind="envelope_parse",
                exit_code=proc.returncode,
                stderr=stderr_text,
                stdout=stdout_text,
            ) from e
        envelope = _result_envelope(parsed)
        if envelope is None:
            blog.error("subscription_envelope_parse", error="no result object", stdout_preview=stdout_text[:500])
            raise self._fail(
                "claude -p returned JSON without a result envelope",
                kind="envelope_parse",
                exit_code=proc.returncode,
                stderr=stderr_text,
                stdout=stdout_text,
            )

        cost = float(envelope.get("total_cost_usd", 0))
        turns = int(envelope.get("num_turns", 0))
        duration_ms = int(envelope.get("duration_ms", 0))
        raw = strip_fences(envelope.get("result", ""))

        try:
            tool_input = json.loads(raw)
        except json.JSONDecodeError as e:
            blog.error("subscription_non_json", raw=raw[:300], stderr=stderr_text[:500])
            raise self._fail(
                f"claude -p result is not JSON: {e}",
                kind="result_parse",
                exit_code=proc.returncode,
                stderr=stderr_text,
                stdout=raw[:_STREAM_CAP],
            ) from e

        try:
            payload = validate_or_coerce(tool_input, req.schema_cls, log_bind=blog, custom_id=req.slug)
        except ValidationError as e:
            blog.error("subscription_schema_violation", errors=str(e)[:300])
            raise self._fail(
                f"schema violation: {e}",
                kind="schema_violation",
                exit_code=proc.returncode,
                stderr=stderr_text,
                stdout=raw[:_STREAM_CAP],
            ) from e

        return InvokeResult(
            payload=payload,
            cost=cost,
            turns=turns,
            duration_ms=duration_ms,
        )
