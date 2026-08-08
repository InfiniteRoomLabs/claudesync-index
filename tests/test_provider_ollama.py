"""Ollama provider: httpx-based /api/chat structured-output path.

Transport mock: patch `reindex.providers.ollama._get_client` to return a
MagicMock whose `.post` and `.get` are AsyncMocks. This mirrors the pattern
in conftest.mock_anthropic_client (monkeypatch on the module-level client
accessor) and is the seam the contract-test fixture will use.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from reindex import config, models
from reindex.providers.base import InvokeRequest, ProviderFailure
from reindex.providers.ollama import OllamaProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _provider() -> OllamaProvider:
    return OllamaProvider(
        config.load(Path("/nonexistent"), provider_name=config.ProviderName.OLLAMA)
    )


def _req(step: str = "leaf") -> InvokeRequest:
    return InvokeRequest(
        step=step,
        slug="ollama-test",
        model="llama3.2",
        system_prompt="Summarize the conversation.",
        user_content="Human: hello\nAssistant: world",
        schema_cls=models.STEP_MODEL[step],
        work_dir=Path("/tmp"),
    )


def _ollama_response(content_dict: dict | None, *, status: int = 200) -> MagicMock:
    """Build a fake httpx.Response for _post_chat.

    `content_dict=None` simulates prose content (non-JSON), as when the model
    ignores the schema and writes an explanation in plain text.
    """
    if content_dict is not None:
        content_str = json.dumps(content_dict)
    else:
        content_str = "Here is a helpful response in plain prose."

    resp = MagicMock()
    resp.status_code = status
    resp.text = json.dumps({
        "message": {"role": "assistant", "content": content_str},
        "prompt_eval_count": 80,
        "eval_count": 40,
    }) if status < 400 else "Internal Server Error"
    resp.json = MagicMock(return_value=json.loads(resp.text) if status < 400 else {})
    return resp


def _make_mock_client(post_side_effect=None, get_response=None):
    """Return a MagicMock httpx.AsyncClient with configurable post/get behaviors.

    `post_side_effect` may be a single response, a list (consumed in order), or
    an exception to raise.
    """
    client = MagicMock()

    if isinstance(post_side_effect, list):
        client.post = AsyncMock(side_effect=post_side_effect)
    elif isinstance(post_side_effect, Exception):
        client.post = AsyncMock(side_effect=post_side_effect)
    else:
        client.post = AsyncMock(return_value=post_side_effect)

    tags_resp = MagicMock()
    tags_resp.status_code = 200
    tags_resp.raise_for_status = MagicMock()
    client.get = AsyncMock(return_value=get_response or tags_resp)
    client.aclose = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path(valid_leaf_dict: dict, monkeypatch: pytest.MonkeyPatch):
    """Valid payload round-trip: tokens mapped, cost is zero (free provider)."""
    mock_client = _make_mock_client(_ollama_response(valid_leaf_dict))
    monkeypatch.setattr("reindex.providers.ollama._get_client", lambda: mock_client)

    result = await _provider().invoke(_req())

    assert isinstance(result.payload, models.LeafSummary)
    assert result.payload.title == valid_leaf_dict["title"]
    # prompt_eval_count=80, eval_count=40 from _ollama_response
    assert result.input_tokens == 80
    assert result.output_tokens == 40
    # Empty pricing table → always free
    assert result.cost == 0.0
    assert result.duration_ms >= 0


# ---------------------------------------------------------------------------
# Non-JSON content → one retry, then result_parse failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prose_content_retries_then_fails(monkeypatch: pytest.MonkeyPatch):
    """When message.content is plain prose (non-JSON), provider retries once.
    Both attempts return prose → ProviderFailure kind=result_parse."""
    mock_client = _make_mock_client([
        _ollama_response(None),   # first call: prose
        _ollama_response(None),   # retry call: prose again
    ])
    monkeypatch.setattr("reindex.providers.ollama._get_client", lambda: mock_client)

    with pytest.raises(ProviderFailure) as exc_info:
        await _provider().invoke(_req())

    assert exc_info.value.kind == "result_parse"
    assert exc_info.value.retryable is True
    # Two POST calls: original + one retry
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_prose_content_retry_succeeds(valid_leaf_dict: dict, monkeypatch: pytest.MonkeyPatch):
    """First call returns prose, retry returns valid JSON → success."""
    mock_client = _make_mock_client([
        _ollama_response(None),            # first: prose
        _ollama_response(valid_leaf_dict), # retry: good JSON
    ])
    monkeypatch.setattr("reindex.providers.ollama._get_client", lambda: mock_client)

    result = await _provider().invoke(_req())

    assert isinstance(result.payload, models.LeafSummary)
    assert mock_client.post.call_count == 2


# ---------------------------------------------------------------------------
# Garbage JSON (wrong schema) → retry, then schema_violation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_schema_garbage_retries_then_fails(monkeypatch: pytest.MonkeyPatch):
    """JSON that doesn't fit LeafSummary and can't be coerced → retry → schema_violation."""
    garbage = {"completely": "wrong", "no": "required fields"}
    mock_client = _make_mock_client([
        _ollama_response(garbage),
        _ollama_response(garbage),
    ])
    monkeypatch.setattr("reindex.providers.ollama._get_client", lambda: mock_client)

    with pytest.raises(ProviderFailure) as exc_info:
        await _provider().invoke(_req())

    assert exc_info.value.kind == "schema_violation"
    assert exc_info.value.retryable is True
    assert mock_client.post.call_count == 2


# ---------------------------------------------------------------------------
# HTTP 500 → http_error, no retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_http_500_raises_http_error(monkeypatch: pytest.MonkeyPatch):
    """HTTP 5xx → ProviderFailure kind=http_error, retryable=False."""
    resp = MagicMock()
    resp.status_code = 500
    resp.text = "Internal Server Error"
    mock_client = _make_mock_client(resp)
    monkeypatch.setattr("reindex.providers.ollama._get_client", lambda: mock_client)

    with pytest.raises(ProviderFailure) as exc_info:
        await _provider().invoke(_req())

    assert exc_info.value.kind == "http_error"
    assert exc_info.value.retryable is False
    # Only one attempt — HTTP errors are not retried in-provider
    assert mock_client.post.call_count == 1


# ---------------------------------------------------------------------------
# Network error → http_error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connect_error_raises_http_error(monkeypatch: pytest.MonkeyPatch):
    """Connection failure → ProviderFailure kind=http_error."""
    mock_client = _make_mock_client(httpx.ConnectError("connection refused"))
    monkeypatch.setattr("reindex.providers.ollama._get_client", lambda: mock_client)

    with pytest.raises(ProviderFailure) as exc_info:
        await _provider().invoke(_req())

    assert exc_info.value.kind == "http_error"
    assert exc_info.value.retryable is False


# ---------------------------------------------------------------------------
# Coercible shape error repairs without retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_coercible_topics_string_repairs_without_retry(
    valid_leaf_dict: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    """string-where-array for `topics` is a mechanical shape error:
    validate_or_coerce repairs it without any retry (one POST call only)."""
    valid_leaf_dict["topics"] = "only-one-topic"  # string, not list
    mock_client = _make_mock_client(_ollama_response(valid_leaf_dict))
    monkeypatch.setattr("reindex.providers.ollama._get_client", lambda: mock_client)

    result = await _provider().invoke(_req())

    assert result.payload.topics == ["only-one-topic"]
    assert mock_client.post.call_count == 1


# ---------------------------------------------------------------------------
# preflight: GET /api/tags failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_preflight_failure_raises_provider_failure(monkeypatch: pytest.MonkeyPatch):
    """preflight() raises ProviderFailure when /api/tags is unreachable."""
    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    monkeypatch.setattr("reindex.providers.ollama._get_client", lambda: mock_client)

    with pytest.raises(ProviderFailure) as exc_info:
        await _provider().preflight()

    assert exc_info.value.kind == "http_error"
    # Message should tell the user how to fix it without private helper references.
    assert "Start the Ollama server" in str(exc_info.value)


@pytest.mark.asyncio
async def test_preflight_success_does_not_raise(monkeypatch: pytest.MonkeyPatch):
    """preflight() completes without exception when /api/tags returns 200."""
    tags_resp = MagicMock()
    tags_resp.status_code = 200
    tags_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=tags_resp)
    monkeypatch.setattr("reindex.providers.ollama._get_client", lambda: mock_client)

    await _provider().preflight()  # must not raise


# ---------------------------------------------------------------------------
# Token mapping
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_zero_token_counts_when_absent(valid_leaf_dict: dict, monkeypatch: pytest.MonkeyPatch):
    """Ollama sometimes omits prompt_eval_count/eval_count (cached responses).
    Provider should default to 0 rather than KeyError."""
    resp = MagicMock()
    resp.status_code = 200
    resp.text = json.dumps({
        "message": {"role": "assistant", "content": json.dumps(valid_leaf_dict)},
        # no prompt_eval_count or eval_count keys
    })
    resp.json = MagicMock(return_value=json.loads(resp.text))
    mock_client = _make_mock_client(resp)
    monkeypatch.setattr("reindex.providers.ollama._get_client", lambda: mock_client)

    result = await _provider().invoke(_req())

    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.cost == 0.0
