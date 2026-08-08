"""Contract tests for the OpenAI-compatible and Ollama embedding backends."""

import httpx
import pytest

from reindex import embedding


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_make_embedder_unknown_backend():
    with pytest.raises(embedding.EmbeddingConfigError, match="cloudflare, ollama, openai"):
        embedding.make_embedder("pinecone", model=None, base_url=None)


def test_make_embedder_cloudflare_missing_creds(monkeypatch):
    monkeypatch.delenv("CF_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CF_API_TOKEN", raising=False)
    with pytest.raises(embedding.EmbeddingConfigError, match="CF_ACCOUNT_ID"):
        embedding.make_embedder("cloudflare", model=None, base_url=None)


def test_make_embedder_defaults_models():
    e = embedding.make_embedder("ollama", model=None, base_url=None)
    assert e.model == embedding.DEFAULT_MODELS["ollama"]
    e2 = embedding.make_embedder("openai", model="custom-model", base_url="http://localhost:8000")
    assert e2.model == "custom-model"


@pytest.mark.asyncio
async def test_openai_embedder_request_shape_and_parse(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        import json
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [
            {"index": 1, "embedding": [0.3, 0.4]},
            {"index": 0, "embedding": [0.1, 0.2]},
        ]})

    monkeypatch.setenv("CSINDEX_EMBED_API_KEY", "sk-test")
    e = embedding.make_embedder("openai", model="m", base_url="http://x")
    e._client = httpx.AsyncClient(transport=_mock_transport(handler))
    vecs = await e.embed(["a", "b"])
    assert captured["url"] == "http://x/v1/embeddings"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"] == {"model": "m", "input": ["a", "b"]}
    assert vecs == [[0.1, 0.2], [0.3, 0.4]]  # re-sorted by index
    await e.aclose()


@pytest.mark.asyncio
async def test_openai_embedder_no_key_no_auth_header(monkeypatch):
    monkeypatch.delenv("CSINDEX_EMBED_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    e = embedding.make_embedder("openai", model="m", base_url="http://local")
    e._client = httpx.AsyncClient(transport=_mock_transport(handler))
    await e.embed(["a"])
    assert seen["auth"] is None
    await e.aclose()


@pytest.mark.asyncio
async def test_ollama_embedder_request_shape_and_parse():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [[0.1], [0.2]]})

    e = embedding.make_embedder("ollama", model=None, base_url=None)
    e._client = httpx.AsyncClient(transport=_mock_transport(handler))
    vecs = await e.embed(["a", "b"])
    assert captured["url"] == "http://127.0.0.1:11434/api/embed"
    assert captured["body"]["input"] == ["a", "b"]
    assert vecs == [[0.1], [0.2]]
    await e.aclose()


@pytest.mark.asyncio
async def test_new_backends_retry_transient_then_succeed():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"embeddings": [[0.5]]})

    e = embedding.make_embedder("ollama", model=None, base_url=None)
    e._client = httpx.AsyncClient(transport=_mock_transport(handler))
    e._retry_base_delay = 0.0
    vecs = await e.embed(["a"])
    assert calls["n"] == 2 and vecs == [[0.5]]
    await e.aclose()


@pytest.mark.asyncio
async def test_new_backends_4xx_fails_immediately():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad"})

    e = embedding.make_embedder("openai", model="m", base_url="http://x")
    e._client = httpx.AsyncClient(transport=_mock_transport(handler))
    with pytest.raises(RuntimeError):
        await e.embed(["a"])
    await e.aclose()
