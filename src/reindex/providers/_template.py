"""Skeleton for a new provider. Copy to <name>.py and work the checklist:

  1. Add the name to config.ProviderName and a defaults block in
     config._BUILTIN_DEFAULTS (models + pricing; an EMPTY pricing dict
     means "free provider" and silences unknown-model warnings).
  2. Implement invoke() below — the only required method.
  3. Map every failure onto ProviderFailure with a stable `kind`
     (process_exit, envelope_parse, result_parse, schema_violation,
     http_error, no_structured_output) and an honest `retryable`:
     "would a stronger model probably fix this?" — drives the runner's
     escalation retry (config.models.escalation; None disables).
  4. Use validate_or_coerce() for payload validation; it auto-repairs
     mechanical shape errors (string-where-array etc.) before failing.
     Consider one in-provider retry with a reminder message before
     raising kind="schema_violation".
  5. Register in providers/__init__.py:_registry().
  6. Add a transport-mock fixture + entry in
     tests/test_provider_contract.py — that one parametrized test is
     what makes a new provider trustworthy.

Implement BatchCapable ONLY if the provider has a real server-side batch
product (its own IDs, polling, result retrieval). Don't fake it with a
loop — the runner's invoke path already is that loop.

Set supports_file_transcripts=True ONLY if invoke() can actually read
work_dir files via a tool (claude-cli's --add-dir Read loop). Otherwise
prepare_leaf inlines all transcripts for you.
"""

from __future__ import annotations

import time

from pydantic import ValidationError

from reindex.providers.base import (
    InvokeRequest,
    InvokeResult,
    Provider,
    ProviderFailure,
    validate_or_coerce,
)


class MyProvider(Provider):
    name = "my-provider"

    async def preflight(self) -> None:
        # Fail fast on missing binaries / API keys. Raise ProviderFailure
        # (kind="process_exit") — the CLI maps it to exit code CONFIG.
        return

    async def invoke(self, req: InvokeRequest) -> InvokeResult:
        t0 = time.monotonic()

        raw = await self._call_transport(req)  # <- your API/subprocess call

        try:
            payload = validate_or_coerce(raw, req.schema_cls, custom_id=req.slug)
        except ValidationError as e:
            raise ProviderFailure(
                f"schema violation: {e}",
                kind="schema_violation",
                provider=self.name,
                retryable=True,
            ) from e

        in_tok, out_tok = 0, 0  # from your transport's usage block, if any
        return InvokeResult(
            payload=payload,
            cost=self.config.compute_cost(req.model, in_tok, out_tok),
            duration_ms=int((time.monotonic() - t0) * 1000),
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    async def _call_transport(self, req: InvokeRequest) -> dict:
        raise NotImplementedError
