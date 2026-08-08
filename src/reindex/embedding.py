"""
Optional full-content vector indexing — an experiment alongside the index.md
hierarchy, NOT a replacement.

Pipeline: walk leaf conversation.md files -> chunk -> embed via Cloudflare
Workers AI (bge-m3) -> upsert into a local persistent Chroma collection.
A query path (search) embeds the query the same way and returns nearest chunks.

This deliberately does NOT reuse the providers/ abstraction: that contract is
LLM-structured-output shaped (text -> Pydantic), whereas embedding is
text -> float vector. Different shape, separate code path, zero blast radius
on the summary pipeline.

Cloudflare creds (environment variables):
    CF_ACCOUNT_ID   Cloudflare account id
    CF_API_TOKEN    API token with the "Workers AI" permission

chromadb is an optional dependency: `uv sync --extra embed`.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from reindex import log, shutdown

# Concurrent files in flight. The bottleneck is the Cloudflare round-trip, so
# parallelism here is pure latency hiding; 8 is well under Workers AI rate caps.
DEFAULT_MAX_IN_FLIGHT = 8

# bge-m3: 60k-token context, 1024-dim dense vectors, $0.012/M input tokens.
DEFAULT_MODEL = "@cf/baai/bge-m3"
# Chunk well below the context window: small chunks retrieve precisely, huge
# chunks blur a whole doc into one averaged vector. ~2000 chars ~= 500 tokens.
DEFAULT_CHUNK_CHARS = 2000
DEFAULT_OVERLAP_CHARS = 200
# Cloudflare's sync embed endpoint bounds a request two ways: at most ~100
# inputs in the array, AND a per-REQUEST total-token cap (bge-m3 = 60k tokens
# across the whole batch, not per input). Pack batches under both. Dense
# markdown/code measured at ~2.5 chars/token, so 60k tokens ~= 150k chars is
# the HARD ceiling; budget at 100k chars (~40k tokens) for comfortable margin.
CF_MAX_ITEMS = 100
CF_MAX_CHARS = 100_000
_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
# Transient failures (connection resets under concurrency, 429s, 5xx) are
# common on bursty API traffic and self-heal on retry. Back off and retry a
# few times before giving up on a batch.
_MAX_RETRIES = 4
_RETRY_BACKOFF_S = 0.5  # doubled each attempt: 0.5, 1, 2, 4


# ---------------------------------------------------------------------------
# Chunking (pure)
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    *,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Fixed-width character chunks with overlap. Char-based, not token-based:
    good enough for a retrieval experiment, no tokenizer dependency.

    Empty/whitespace-only input -> no chunks. Overlap must be < max_chars.
    """
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    step = max_chars - overlap
    out: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + max_chars].strip()
        if piece:
            out.append(piece)
        if start + max_chars >= len(text):
            break
    return out


# ---------------------------------------------------------------------------
# Cloudflare Workers AI embedder
# ---------------------------------------------------------------------------

@dataclass
class CFConfig:
    account_id: str
    api_token: str
    model: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls, model: str = DEFAULT_MODEL) -> CFConfig:
        acct = os.environ.get("CF_ACCOUNT_ID", "").strip()
        token = os.environ.get("CF_API_TOKEN", "").strip()
        missing = [n for n, v in (("CF_ACCOUNT_ID", acct), ("CF_API_TOKEN", token)) if not v]
        if missing:
            raise RuntimeError(
                f"Missing Cloudflare creds: {', '.join(missing)}. "
                "Set them in the environment (token needs the 'Workers AI' permission)."
            )
        return cls(account_id=acct, api_token=token, model=model)

    @property
    def url(self) -> str:
        return (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self.account_id}/ai/run/{self.model}"
        )


class _TransientError(Exception):
    """Internal marker: a retryable response (429/5xx). Never escapes embed()."""


def _pack_batches(
    texts: list[str],
    *,
    max_items: int = CF_MAX_ITEMS,
    max_chars: int = CF_MAX_CHARS,
) -> list[list[str]]:
    """Greedily group texts into batches bounded by item count AND total chars
    (proxy for CF's per-request token cap). A single text larger than max_chars
    still goes out alone — chunk_text keeps chunks well under the limit, so this
    is just a safety valve, not silent truncation."""
    batches: list[list[str]] = []
    cur: list[str] = []
    cur_chars = 0
    for t in texts:
        if cur and (len(cur) >= max_items or cur_chars + len(t) > max_chars):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(t)
        cur_chars += len(t)
    if cur:
        batches.append(cur)
    return batches


def _extract_vectors(result: Any) -> list[list[float]]:
    """Pull the list-of-dense-vectors out of Cloudflare's result object.

    CF hasn't published bge-m3's exact response shape, and it has drifted
    across bge models, so handle the known variants and fail LOUD on anything
    else (parsing an API response is a trust boundary — never guess silently):
      {"data": [[...], ...]}                      classic bge-small/base/large
      {"response": [[...], ...]}                  some newer models
      {"response": [{"embedding"|"dense_vec": [...]}, ...]}  object-wrapped
    """
    if not isinstance(result, dict):
        raise ValueError(f"unexpected CF result type: {type(result).__name__}")

    raw = result.get("data")
    if raw is None:
        raw = result.get("response")
    if raw is None:
        raise ValueError(f"no 'data'/'response' in CF result; keys={list(result.keys())}")

    vecs: list[list[float]] = []
    for item in raw:
        if isinstance(item, dict):
            v = item.get("embedding") or item.get("dense_vec") or item.get("dense")
            if v is None:
                raise ValueError(f"vector dict missing embedding field; keys={list(item.keys())}")
            vecs.append([float(x) for x in v])
        else:
            vecs.append([float(x) for x in item])
    return vecs


class CloudflareEmbedder:
    def __init__(self, cfg: CFConfig) -> None:
        self.cfg = cfg
        self._client = httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers={"Authorization": f"Bearer {cfg.api_token}"},
        )
        self._log = log.get("embed.cf")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, packing batches under CF's per-request item and token
        caps. Order preserved. Transient errors are retried with backoff."""
        out: list[list[float]] = []
        for batch in _pack_batches(texts):
            vecs = await self._embed_batch(batch)
            out.extend(vecs)
            self._log.debug("cf_batch", n=len(batch), dim=len(vecs[0]) if vecs else 0)
        return out

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """One CF request with bounded retry on transient failures. A 4xx
        (other than 429) is a real client error and fails immediately — only
        connection errors, timeouts, 429, and 5xx are retried."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await self._client.post(self.cfg.url, json={"text": batch})
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise _TransientError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"Cloudflare embed HTTP {resp.status_code}: {resp.text[:500]}"
                    )
                body = resp.json()
                if not body.get("success", True):
                    raise RuntimeError(f"Cloudflare embed error: {body.get('errors')}")
                vecs = _extract_vectors(body.get("result", body))
                if len(vecs) != len(batch):
                    raise RuntimeError(
                        f"CF returned {len(vecs)} vectors for {len(batch)} inputs"
                    )
                return vecs
            except (_TransientError, httpx.TransportError, httpx.TimeoutException) as e:
                last_exc = e
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BACKOFF_S * (2 ** attempt)
                    self._log.warning(
                        "cf_retry", attempt=attempt + 1, delay=delay,
                        error_type=type(e).__name__, error=str(e)[:200],
                    )
                    await asyncio.sleep(delay)
        raise RuntimeError(
            f"Cloudflare embed failed after {_MAX_RETRIES} attempts: "
            f"{type(last_exc).__name__}: {last_exc}"
        )

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Corpus walk + Chroma store
# ---------------------------------------------------------------------------

# Two kinds of source share the store, in separate id namespaces so they never
# collide and conversation ids stay backward-compatible (`slug#i`):
#   conversation — raw conversation.md (precise, verbose)
#   summary      — generated INDEX.md (concept-dense, vocabulary-normalized;
#                  this is what closes the paraphrase gap in retrieval)
KIND_CONVERSATION = "conversation"
KIND_SUMMARY = "summary"


@dataclass
class Source:
    path: Path
    slug: str   # directory of the file, relative to the export root
    kind: str

    @property
    def id_prefix(self) -> str:
        # Conversations keep the legacy bare-slug namespace; summaries get their
        # own suffix so both can describe the same directory without colliding.
        return self.slug if self.kind == KIND_CONVERSATION else f"{self.slug}::{self.kind}"


def _leaf_dirs(root: Path) -> Iterator[Path]:
    conv = root / "conversations"
    if conv.is_dir():
        yield from sorted(p for p in conv.iterdir() if p.is_dir())
    projects = root / "projects"
    if projects.is_dir():
        for proj in sorted(p for p in projects.iterdir() if p.is_dir()):
            c = proj / "conversations"
            if c.is_dir():
                yield from sorted(p for p in c.iterdir() if p.is_dir())


def iter_sources(
    root: Path, *, conversations: bool = True, summaries: bool = True,
) -> Iterator[Source]:
    """Yield embeddable Sources. `conversations` = raw conversation.md at each
    leaf. `summaries` = generated INDEX.md at every level (leaf, project, root)
    — the concept-dense layer agentic search beat us with."""
    def src(f: Path, kind: str) -> Source:
        return Source(path=f, slug=str(f.parent.relative_to(root)), kind=kind)

    for d in _leaf_dirs(root):
        if conversations and (f := d / "conversation.md").is_file():
            yield src(f, KIND_CONVERSATION)
        if summaries and (f := d / "INDEX.md").is_file():
            yield src(f, KIND_SUMMARY)

    if summaries:
        # Project-aggregate and root INDEX.md — no conversation.md counterpart.
        projects = root / "projects"
        if projects.is_dir():
            for proj in sorted(p for p in projects.iterdir() if p.is_dir()):
                if (f := proj / "INDEX.md").is_file():
                    yield src(f, KIND_SUMMARY)
        if (f := root / "INDEX.md").is_file():
            yield src(f, KIND_SUMMARY)


def get_collection(persist_dir: Path, name: str = "conversations"):
    """Local persistent Chroma collection, cosine space. Imported lazily so
    the rest of the package doesn't hard-depend on chromadb."""
    try:
        import chromadb  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "chromadb not installed. Install the embed extra: uv sync --extra embed "
            "(or pip install 'claudesync-index[embed]')."
        ) from e
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


@dataclass
class EmbedStats:
    files: int = 0
    chunks: int = 0
    skipped: int = 0  # unchanged since last run (content-hash hit)
    failed: int = 0   # per-file errors that did NOT abort the run


def _content_hash(text: str, *, max_chars: int, overlap: int) -> str:
    """Cache key for a file. Includes chunk params so changing how we chunk
    busts the cache (re-embed), mirroring the schema-hash idea in the LLM
    pipeline's content_hash."""
    h = hashlib.sha256()
    h.update(f"{max_chars}:{overlap}:".encode())
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _stored_hash(col, id_prefix: str) -> str | None:
    """The content_hash recorded on this source's first chunk, or None if absent."""
    got = col.get(ids=[f"{id_prefix}#0"], include=["metadatas"])
    metas = got.get("metadatas") or []
    if metas:
        return metas[0].get("content_hash")
    return None


def backfill_kind(persist_dir: Path) -> int:
    """One-shot, idempotent migration: tag pre-`kind` entries (which were all
    conversations) with kind="conversation". Metadata-only update — reuses the
    stored vectors, so NO re-embedding and no Cloudflare cost. Returns the count
    of chunks updated. Safe to run repeatedly: already-tagged rows are skipped.
    """
    col = get_collection(persist_dir)
    got = col.get(include=["metadatas"])
    ids = got.get("ids") or []
    metas = got.get("metadatas") or []
    fix_ids, fix_metas = [], []
    for cid, meta in zip(ids, metas, strict=True):
        if meta.get("kind") is None:
            fix_ids.append(cid)
            fix_metas.append({**meta, "kind": KIND_CONVERSATION})
    # Chroma caps a single update at ~5461 rows; chunk to stay under it.
    step = 5000
    for i in range(0, len(fix_ids), step):
        col.update(ids=fix_ids[i : i + step], metadatas=fix_metas[i : i + step])
    return len(fix_ids)


async def embed_corpus(
    root: Path,
    persist_dir: Path,
    *,
    embedder: CloudflareEmbedder,
    limit: int = 0,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
    force: bool = False,
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
    conversations: bool = True,
    summaries: bool = True,
) -> EmbedStats:
    """Walk sources, chunk, embed via Cloudflare, store into Chroma.

    Sources are of two `kind`s — raw conversation.md and generated INDEX.md
    summaries — in separate id namespaces (conversation = `slug#i`, summary =
    `slug::summary#i`), so a directory's conversation and summary coexist.

    Concurrency: up to `max_in_flight` sources embed in parallel (the Cloudflare
    round-trip is the bottleneck). Chroma mutations are funnelled through one
    lock — its PersistentClient is not safe for concurrent writers.

    Shutdown: cooperative. On Ctrl-C the controller cancels in-flight tasks; we
    stop scheduling. Each source commits atomically, so a cancelled run leaves a
    clean partial DB the content-hash skip resumes from — no orphan/half state.

    Idempotent like the index.md pipeline:
      * content-hash skip — unchanged sources (content + chunk params) cost $0;
      * clean replace — changed sources drop prior chunks for their (slug, kind)
        before writing, so a shrunk file leaves no orphans. `force` busts it.
    """
    col = get_collection(persist_dir)
    olog = log.get("embed")
    stats = EmbedStats()
    controller = shutdown.get()

    sources = list(iter_sources(root, conversations=conversations, summaries=summaries))
    if limit > 0:
        sources = sources[:limit]
    olog.info("embed_started", sources=len(sources), persist=str(persist_dir),
              force=force, max_in_flight=max_in_flight,
              conversations=conversations, summaries=summaries)

    sem = asyncio.Semaphore(max_in_flight)
    write_lock = asyncio.Lock()

    async def handle(src: Source) -> None:
        async with sem:
            if controller is not None and controller.is_shutting_down():
                return
            try:
                text = src.path.read_text(encoding="utf-8")
                chash = _content_hash(text, max_chars=max_chars, overlap=overlap)

                async with write_lock:
                    stored = None if force else _stored_hash(col, src.id_prefix)
                if stored == chash and not force:
                    stats.skipped += 1
                    olog.debug("embed_skip_unchanged", slug=src.slug, kind=src.kind)
                    return

                chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
                if not chunks:
                    return
                vectors = await embedder.embed(chunks)

                # Clean replace under the lock: drop prior chunks for this
                # (slug, kind) — kind-scoped so we never clobber the sibling
                # kind — then write the fresh set with content_hash stamped.
                async with write_lock:
                    col.delete(where={"$and": [{"slug": src.slug}, {"kind": src.kind}]})
                    col.add(
                        ids=[f"{src.id_prefix}#{i}" for i in range(len(chunks))],
                        embeddings=vectors,  # type: ignore[arg-type]  # chroma stub wants ndarray; list works
                        documents=chunks,
                        metadatas=[
                            {"slug": src.slug, "kind": src.kind, "path": str(src.path),
                             "chunk": i, "content_hash": chash}
                            for i in range(len(chunks))
                        ],
                    )
                stats.files += 1
                stats.chunks += len(chunks)
                olog.info("embedded", slug=src.slug, kind=src.kind, chunks=len(chunks))
            except asyncio.CancelledError:
                raise  # shutdown path — let it propagate to cancel the gather
            except Exception as e:
                # Per-source isolation: one bad file (oversize chunk, transient
                # HTTP error, encoding issue) must not abort the run. The
                # content-hash skip retries only what didn't land next time.
                stats.failed += 1
                olog.error("embed_file_failed", slug=src.slug, kind=src.kind,
                           error_type=type(e).__name__, error=str(e)[:300])

    tasks = [asyncio.create_task(handle(s)) for s in sources]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        olog.warn("embed_interrupted", files=stats.files, chunks=stats.chunks,
                  skipped=stats.skipped, failed=stats.failed)
        raise
    finally:
        olog.info(
            "embed_done",
            files=stats.files, chunks=stats.chunks,
            skipped=stats.skipped, failed=stats.failed,
        )
    return stats


async def search(
    persist_dir: Path,
    query: str,
    *,
    embedder: CloudflareEmbedder,
    k: int = 5,
    kind: str | None = None,
) -> list[dict]:
    """Embed the query and return the k nearest chunks (text + metadata + score).
    `kind` filters to "conversation" or "summary"; None searches both."""
    col = get_collection(persist_dir)
    qvec = (await embedder.embed([query]))[0]
    where = {"kind": kind} if kind else None
    res = col.query(query_embeddings=[qvec], n_results=k, where=where)
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    hits = []
    for doc, meta, dist in zip(docs, metas, dists, strict=True):
        hits.append({"score": 1.0 - dist, "slug": meta.get("slug"),
                     "kind": meta.get("kind"), "text": doc, "meta": meta})
    return hits
