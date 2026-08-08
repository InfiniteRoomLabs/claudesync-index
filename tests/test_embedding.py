"""Unit tests for the pure embedding logic: chunking + CF response parsing.

Plus a set of integration-shaped tests below (`get_collection`, `iter_sources`,
`embed_corpus`, `search`, `backfill_kind`) that exercise the real (local,
on-disk) chromadb path with a fake, network-free embedder -- the CF HTTP path
itself stays covered by the pure-logic tests above.
"""

from pathlib import Path

import httpx
import pytest

from reindex import embedding
from reindex.embedding import (
    KIND_CONVERSATION,
    KIND_SUMMARY,
    CFConfig,
    CloudflareEmbedder,
    Source,
    _content_hash,
    _extract_vectors,
    _pack_batches,
    chunk_text,
)


def test_source_id_namespaces_do_not_collide():
    slug = "conversations/foo"
    conv = Source(path=Path("x"), slug=slug, kind=KIND_CONVERSATION)
    summ = Source(path=Path("y"), slug=slug, kind=KIND_SUMMARY)
    # conversation keeps the legacy bare-slug namespace (backward compatible)
    assert conv.id_prefix == slug
    # summary gets its own namespace so same-dir sources never share an id
    assert summ.id_prefix == f"{slug}::summary"
    assert conv.id_prefix != summ.id_prefix


class _FakeResp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _embedder(monkeypatch, responses):
    """A CloudflareEmbedder whose POST yields the given responses/exceptions
    in order. Backoff sleeps are stubbed to keep the test instant."""
    monkeypatch.setattr("reindex.embedding.asyncio.sleep", _noop_sleep)
    emb = CloudflareEmbedder(CFConfig(account_id="a", api_token="t"))
    seq = iter(responses)

    async def fake_post(url, json):
        item = next(seq)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(emb._client, "post", fake_post)
    return emb


async def _noop_sleep(_):
    return None


OK_BODY = {"success": True, "result": {"data": [[1.0, 2.0]]}}


def test_chunk_short_text_single_chunk():
    assert chunk_text("hello world", max_chars=100, overlap=10) == ["hello world"]


def test_chunk_empty_is_no_chunks():
    assert chunk_text("   \n  ") == []


def test_chunk_splits_with_overlap_and_covers_all_text():
    text = "abcdefghij" * 10  # 100 chars
    chunks = chunk_text(text, max_chars=40, overlap=10)
    assert len(chunks) > 1
    # overlap means consecutive chunks share a tail/head
    assert chunks[0][-10:] == chunks[1][:10]
    # last char of source appears in the final chunk (full coverage)
    assert text[-1] in chunks[-1]


def test_chunk_rejects_overlap_ge_maxchars():
    with pytest.raises(ValueError):
        chunk_text("x" * 50, max_chars=20, overlap=20)


def test_extract_vectors_data_shape():
    assert _extract_vectors({"data": [[1, 2], [3, 4]]}) == [[1.0, 2.0], [3.0, 4.0]]


def test_extract_vectors_response_shape():
    assert _extract_vectors({"response": [[0.5]]}) == [[0.5]]


def test_extract_vectors_object_wrapped():
    out = _extract_vectors({"response": [{"dense_vec": [1, 2]}, {"embedding": [3, 4]}]})
    assert out == [[1.0, 2.0], [3.0, 4.0]]


def test_extract_vectors_unknown_shape_fails_loud():
    with pytest.raises(ValueError):
        _extract_vectors({"weird": 1})


def test_pack_batches_respects_item_cap():
    batches = _pack_batches(["x"] * 250, max_items=100, max_chars=10**9)
    assert [len(b) for b in batches] == [100, 100, 50]


def test_pack_batches_respects_char_cap():
    # each text 40 chars, cap 100 -> 2 per batch
    batches = _pack_batches(["a" * 40] * 5, max_items=1000, max_chars=100)
    assert [len(b) for b in batches] == [2, 2, 1]
    # every text preserved, order intact
    assert sum(len(b) for b in batches) == 5


def test_pack_batches_oversize_single_goes_alone():
    batches = _pack_batches(["small", "X" * 500, "small"], max_items=100, max_chars=100)
    assert ["X" * 500] in batches


def test_content_hash_stable_and_param_sensitive():
    a = _content_hash("hello", max_chars=2000, overlap=200)
    assert a == _content_hash("hello", max_chars=2000, overlap=200)  # deterministic
    assert a != _content_hash("hellp", max_chars=2000, overlap=200)  # content busts
    assert a != _content_hash("hello", max_chars=1000, overlap=200)  # chunk param busts


async def test_embed_retries_transient_then_succeeds(monkeypatch):
    # connection reset, then 503, then OK -> should recover and return vectors
    emb = _embedder(monkeypatch, [
        httpx.ConnectError("reset"),
        _FakeResp(503, text="bad gateway"),
        _FakeResp(200, OK_BODY),
    ])
    out = await emb.embed(["one chunk"])
    assert out == [[1.0, 2.0]]


async def test_embed_gives_up_after_max_retries(monkeypatch):
    emb = _embedder(monkeypatch, [httpx.ConnectError("reset")] * 10)
    with pytest.raises(RuntimeError, match="after .* attempts"):
        await emb.embed(["one chunk"])


async def test_embed_4xx_fails_immediately_no_retry(monkeypatch):
    # a 400 is a real client error: must NOT be retried (would waste the budget)
    emb = _embedder(monkeypatch, [_FakeResp(400, text="bad request")])
    with pytest.raises(RuntimeError, match="HTTP 400"):
        await emb.embed(["one chunk"])


# ---------------------------------------------------------------------------
# CFConfig.from_env
# ---------------------------------------------------------------------------

def test_cfconfig_from_env_success(monkeypatch):
    monkeypatch.setenv("CF_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("CF_API_TOKEN", "tok456")
    cfg = CFConfig.from_env()
    assert cfg.account_id == "acct123"
    assert cfg.api_token == "tok456"
    assert cfg.url.endswith("/accounts/acct123/ai/run/@cf/baai/bge-m3")


def test_cfconfig_from_env_missing_raises(monkeypatch):
    monkeypatch.delenv("CF_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CF_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="Missing Cloudflare creds"):
        CFConfig.from_env()


# ---------------------------------------------------------------------------
# get_collection / iter_sources -- real on-disk chromadb, no network
# ---------------------------------------------------------------------------

def test_get_collection_creates_persistent_store(tmp_path):
    col = embedding.get_collection(tmp_path / "vdb")
    assert col.name == "conversations"


def test_iter_sources_walks_standalone_project_and_root(tmp_export, make_conv, make_project):
    make_conv("alpha")
    (tmp_export / "conversations" / "alpha" / "INDEX.md").write_text("summary alpha", encoding="utf-8")
    proj = make_project("proj1")
    (proj / "INDEX.md").write_text("project summary", encoding="utf-8")
    (tmp_export / "INDEX.md").write_text("root summary", encoding="utf-8")

    sources = list(embedding.iter_sources(tmp_export))
    kinds = {(s.slug, s.kind) for s in sources}
    assert ("conversations/alpha", KIND_CONVERSATION) in kinds
    assert ("conversations/alpha", KIND_SUMMARY) in kinds
    assert ("projects/proj1", KIND_SUMMARY) in kinds
    assert (".", KIND_SUMMARY) in kinds


def test_iter_sources_respects_conversations_and_summaries_flags(tmp_export, make_conv):
    make_conv("alpha")
    (tmp_export / "conversations" / "alpha" / "INDEX.md").write_text("summary alpha", encoding="utf-8")

    conv_only = list(embedding.iter_sources(tmp_export, summaries=False))
    assert {s.kind for s in conv_only} == {KIND_CONVERSATION}

    summ_only = list(embedding.iter_sources(tmp_export, conversations=False))
    assert {s.kind for s in summ_only} == {KIND_SUMMARY}


# ---------------------------------------------------------------------------
# embed_corpus / search / backfill_kind -- fake embedder, real local chromadb
# ---------------------------------------------------------------------------

class _FakeEmbedder(CloudflareEmbedder):
    """Network-free stand-in for CloudflareEmbedder: deterministic vectors
    keyed off text content, so distinct chunks land at distinct points.
    Subclasses (rather than duck-types) CloudflareEmbedder purely so it
    satisfies embed_corpus/search's type signature; the real __init__
    (which opens an httpx client) is deliberately never called."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float((hash(t) % 97) + i) for i in range(self.dim)] for t in texts]


async def test_embed_corpus_skips_unchanged_and_search_finds_it(tmp_export, make_conv, tmp_path):
    make_conv("alpha", "This is a test conversation about widgets and gadgets.")
    persist_dir = tmp_path / "vdb"
    embedder = _FakeEmbedder()

    stats = await embedding.embed_corpus(tmp_export, persist_dir, embedder=embedder, summaries=False)
    assert stats.files == 1
    assert stats.chunks >= 1
    assert stats.skipped == 0
    assert stats.failed == 0

    # Unchanged content on a second pass -> content-hash skip, no re-embed.
    stats2 = await embedding.embed_corpus(tmp_export, persist_dir, embedder=embedder, summaries=False)
    assert stats2.files == 0
    assert stats2.skipped == 1

    # force=True busts the cache even though content is unchanged.
    stats3 = await embedding.embed_corpus(
        tmp_export, persist_dir, embedder=embedder, summaries=False, force=True,
    )
    assert stats3.files == 1
    assert stats3.skipped == 0

    hits = await embedding.search(persist_dir, "widgets", embedder=embedder, k=1)
    assert len(hits) == 1
    assert hits[0]["slug"] == "conversations/alpha"
    assert hits[0]["kind"] == KIND_CONVERSATION

    # kind filter excludes it (no summary source was embedded).
    no_hits = await embedding.search(persist_dir, "widgets", embedder=embedder, k=1, kind=KIND_SUMMARY)
    assert no_hits == []


async def test_embed_corpus_limit_and_empty_root_are_no_ops(tmp_export, tmp_path):
    persist_dir = tmp_path / "vdb"
    embedder = _FakeEmbedder()
    stats = await embedding.embed_corpus(tmp_export, persist_dir, embedder=embedder)
    assert stats == embedding.EmbedStats()


def test_backfill_kind_tags_pre_kind_rows_and_is_idempotent(tmp_path):
    persist_dir = tmp_path / "vdb"
    col = embedding.get_collection(persist_dir)
    col.add(
        ids=["conversations/alpha#0"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["doc"],
        metadatas=[{"slug": "conversations/alpha"}],
    )

    n = embedding.backfill_kind(persist_dir)
    assert n == 1
    got = col.get(ids=["conversations/alpha#0"], include=["metadatas"])
    assert got["metadatas"][0]["kind"] == KIND_CONVERSATION

    # Idempotent: already-tagged rows are left alone on a second pass.
    assert embedding.backfill_kind(persist_dir) == 0
