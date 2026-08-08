"""
claude-cli provider: `claude -p` subprocess (subscription quota).

The only provider with supports_file_transcripts — oversized transcripts
are delivered via the Read tool with --add-dir instead of inlined.
Structured output via --json-schema plus a local validate/coerce pass.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil

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

        req.allow_filesystem: when True, whitelist req.work_dir via
        --add-dir so the model can call the Read tool on files inside it.
        Used by the file-mode leaf summarizer for transcripts too large to
        inline. Write/Edit/NotebookEdit/Bash stay disallowed in both modes.
        """
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

        # Lock the subprocess down to JSON-out only:
        #   * --disallowed-tools removes Write/Edit/NotebookEdit/Bash so the model
        #     can't write a side-effect artifact and return prose about it instead
        #     of the required JSON payload (observed failure mode: the summarizer
        #     misread the request as an implementation task, wrote a file to disk,
        #     and returned a status message instead of structured output).
        #   * --add-dir only when allow_filesystem=True (file-mode leaves). The
        #     inline path doesn't need filesystem access; the transcript is in
        #     stdin via user_content.
        #   * permission-mode left at default (was acceptEdits). With no
        #     side-effect tools allowed, the mode is moot, but 'default' is the
        #     more honest baseline.
        cli_args = [
            "claude", "-p",
            "--model", req.model,
            "--disallowed-tools", "Write Edit NotebookEdit Bash",
            "--output-format", "json",
            "--json-schema", schema_str,
        ]
        if req.allow_filesystem and req.work_dir is not None:
            cli_args.extend(["--add-dir", str(req.work_dir)])

        proc = await asyncio.create_subprocess_exec(
            *cli_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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
            raise self._fail(
                f"claude -p exit {proc.returncode}",
                kind="process_exit",
                exit_code=proc.returncode,
                stderr=stderr_text,
                stdout=stdout_text,
            )

        try:
            envelope = json.loads(stdout.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            blog.error("subscription_envelope_parse", error=str(e)[:300], stdout_preview=stdout_text[:500])
            raise self._fail(
                f"claude -p returned non-JSON envelope: {e}",
                kind="envelope_parse",
                exit_code=proc.returncode,
                stderr=stderr_text,
                stdout=stdout_text,
            ) from e

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
