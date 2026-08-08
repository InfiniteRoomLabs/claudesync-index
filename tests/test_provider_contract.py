"""Provider contract test — parametrized over the registry.

Every registered provider must, with its transport mocked:
  * return an InvokeResult whose payload is the validated schema instance
  * raise ProviderFailure with a stable non-empty `kind` on garbage output
  * report cost >= 0

A new provider gets meaningful coverage from one fixture entry in
_TRANSPORTS below. Deeper provider-specific behavior (envelope parsing,
batch mechanics) belongs in its own test file.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reindex import config
from reindex.models import LeafSummary
from reindex.providers import _registry
from reindex.providers.base import InvokeRequest, InvokeResult, ProviderFailure


def _req(schema_cls=LeafSummary, model: str = "test-model") -> InvokeRequest:
    return InvokeRequest(
        step="leaf", slug="contract", model=model,
        system_prompt="sys", user_content="user",
        schema_cls=schema_cls, work_dir=Path("/tmp"),
    )


# ---------------------------------------------------------------------------
# Per-provider transport mocks: name -> (good_ctx, garbage_ctx)
# Each entry is a callable(payload_dict | None) returning a context manager
# that fakes the provider's transport. None => produce schema-garbage.
# ---------------------------------------------------------------------------

def _claude_cli_transport(payload: dict | None):
    if payload is not None:
        result = json.dumps(payload)
    else:
        result = json.dumps({"not": "the schema"})
    envelope = json.dumps({
        "total_cost_usd": 0.01, "num_turns": 1, "duration_ms": 50,
        "result": result,
    }).encode()

    class _FakeProc:
        returncode = 0

        async def communicate(self, input=None):
            return envelope, b""

    return patch(
        "reindex.providers.claude_cli.asyncio.create_subprocess_exec",
        AsyncMock(return_value=_FakeProc()),
    )


def _anthropic_transport(payload: dict | None):
    block = MagicMock()
    block.type = "tool_use"
    block.input = payload if payload is not None else {"not": "the schema"}
    message = MagicMock()
    message.content = [block]
    message.usage.input_tokens = 100
    message.usage.output_tokens = 50

    client = MagicMock()
    client.messages.create = AsyncMock(return_value=message)
    return patch(
        "reindex.providers.anthropic_api.get_async_client",
        return_value=client,
    )


def _ollama_transport(payload: dict | None):
    content = json.dumps(payload if payload is not None else {"not": "the schema"})
    resp = MagicMock()
    resp.status_code = 200
    resp.text = json.dumps({
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 80,
        "eval_count": 40,
    })
    resp.json = MagicMock(return_value=json.loads(resp.text))

    client = MagicMock()
    # Same response on the provider's internal schema-violation retry.
    client.post = AsyncMock(return_value=resp)
    return patch("reindex.providers.ollama._get_client", return_value=client)


def _opencode_transport(payload: dict | None):
    text = json.dumps(payload if payload is not None else {"not": "the schema"})
    stream = "\n".join([
        json.dumps({"type": "step_start", "part": {"type": "step-start"}}),
        json.dumps({"type": "text", "part": {"type": "text", "text": text}}),
        json.dumps({"type": "step_finish", "part": {
            "type": "step-finish",
            "tokens": {"total": 100, "input": 80, "output": 20,
                       "reasoning": 0, "cache": {"write": 0, "read": 0}},
            "cost": 0.0004,
        }}),
    ]).encode()

    class _FakeProc:
        returncode = 0

        async def communicate(self, input=None):
            return stream, b""

    return patch(
        "reindex.providers.opencode.asyncio.create_subprocess_exec",
        AsyncMock(return_value=_FakeProc()),
    )


_TRANSPORTS = {
    config.ProviderName.CLAUDE_CLI: _claude_cli_transport,
    config.ProviderName.ANTHROPIC: _anthropic_transport,
    config.ProviderName.OLLAMA: _ollama_transport,
    config.ProviderName.OPENCODE: _opencode_transport,
}


def _providers():
    reg = _registry()
    missing = set(reg) - set(_TRANSPORTS)
    assert not missing, f"providers missing a contract transport mock: {missing}"
    return [(name, cls) for name, cls in sorted(reg.items())]


@pytest.mark.parametrize("name,cls", _providers(), ids=lambda v: str(v))
@pytest.mark.asyncio
async def test_valid_payload_roundtrip(name, cls, valid_leaf_dict):
    provider = cls(config.load(Path("/nonexistent"), provider_name=name))
    with _TRANSPORTS[name](valid_leaf_dict):
        result = await provider.invoke(_req())
    assert isinstance(result, InvokeResult)
    assert isinstance(result.payload, LeafSummary)
    assert result.payload.title == valid_leaf_dict["title"]
    assert result.cost >= 0


@pytest.mark.parametrize("name,cls", _providers(), ids=lambda v: str(v))
@pytest.mark.asyncio
async def test_garbage_output_raises_provider_failure(name, cls):
    provider = cls(config.load(Path("/nonexistent"), provider_name=name))
    with _TRANSPORTS[name](None):
        with pytest.raises(ProviderFailure) as exc_info:
            await provider.invoke(_req())
    assert exc_info.value.kind
    assert exc_info.value.provider == name
