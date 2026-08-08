# Phase 2: Restore `quick` and `embed`/`search`/`embed-migrate` — Design

**Date:** 2026-08-08
**Branch:** `feature/phase2-cut-surface`
**Status:** Approved pending user spec review

## Goal

Restore two of the three surfaces cut from the public v1 (`quick`, and the embedding trio `embed`/`search`/`embed-migrate`), generalized so they work for any user on any machine. The `work` sub-app stays out (its value is coupled to a private agent workflow). Upstream source for the restored code is the private repo's `embedding.py`, the `quick` command in its `cli.py`, and its `prompts/reindex.md`.

## Non-goals

- The `work` sub-app.
- Making the private repo consume this package (needs a published release first; separate effort).
- Any change to the summary-pipeline `providers/` abstraction. Embedding stays a separate, deliberately different-shaped seam (text -> vector, not text -> Pydantic).

## Architecture

### Embedding backends

`embedding.py` already passes an `embedder` object into `embed_corpus()`/`search()`. Formalize that seam in place (no new subpackage):

- `class Embedder(Protocol)`: `async def embed(self, texts: list[str]) -> list[list[float]]`, `async def aclose(self) -> None`, and a readonly `model: str` attribute.
- Three implementations in `embedding.py`:
  - `CloudflareEmbedder` — existing code, unchanged mechanics. Workers AI REST, batching, retry on transient HTTP.
  - `OpenAIEmbedder` — POST `{base_url}/v1/embeddings` with `{"model": ..., "input": [...]}`, `Authorization: Bearer` when a key is present. Covers OpenAI, LM Studio, vLLM, and any compatible server.
  - `OllamaEmbedder` — POST `{base_url}/api/embed` (native Ollama API, `{"model": ..., "input": [...]}`). No auth.
- `BACKENDS: dict[str, ...]` name -> factory. Factories raise `EmbeddingConfigError` (caught at the CLI edge -> exit 78 CONFIG) when required settings are missing.
- All three map transport failures to the existing `_TransientError` retry path; non-transient errors keep the existing per-source isolation in `embed_corpus` (one bad file never aborts the run).

### Configuration

New `[embedding]` table in `reindex.toml`, parsed in `config.py` alongside the existing provider config, same precedence discipline as the summary side (CLI flag > env > toml > built-in default):

| Setting | toml key | Env | CLI flag | Default |
|---|---|---|---|---|
| Backend | `embedding.backend` | `CSINDEX_EMBED_BACKEND` | `--backend` | none — required; missing -> exit 78 with a pick-a-backend message naming all three |
| Model | `embedding.model` | `CSINDEX_EMBED_MODEL` | `--model` | per backend: `@cf/baai/bge-m3` / `text-embedding-3-small` / `nomic-embed-text` |
| Base URL | `embedding.base_url` | `CSINDEX_EMBED_BASE_URL` | `--base-url` | openai: `https://api.openai.com`; ollama: `http://127.0.0.1:11434`; cloudflare: n/a (account-derived) |

Credentials are env-only, never toml: `CF_ACCOUNT_ID` + `CF_API_TOKEN` (cloudflare), `CSINDEX_EMBED_API_KEY` falling back to `OPENAI_API_KEY` (openai; optional for local servers), none (ollama).

### Collection compatibility guard

Chroma collections are dimension-bound; querying a bge-m3 (1024-dim) collection with nomic-embed-text (768-dim) vectors fails or, worse, silently degrades across models of equal dimension. Guard:

- On collection creation, stamp `{"embed_backend": ..., "embed_model": ...}` into collection metadata.
- On every `embed`/`search` open, compare stamped backend+model to the active config. Mismatch (or a pre-existing unstamped collection) -> exit 65 DATAERR with one remedy: delete the persist dir and re-embed. Switching backends/models costs a full re-embed, by design — this ships as a "you know what you're getting into" tool, and the guard exists only to make the failure loud instead of silently corrupt. `embed-migrate` needs no embedder (pure Chroma metadata backfill), so it neither requires backend config nor runs the guard.

### `quick`

Restored as-is in behavior: pipe a prompt into `claude -p` under the export-root advisory lock. Generalization:

- The prompt ships as package data `src/reindex/prompts/quick.md` (adapted from the private `reindex.md`, provenance-scrubbed), loaded via `prompt_loader.load_prompt("quick")` so `--prompts-dir` overrides it per-file like every other prompt.
- `{{EXPORT_DIR}}` substitution keyed off the resolved export root (Task 3 semantics: `--root`/`$CSINDEX_ROOT`/CWD, exit 65 on invalid tree).
- Restore `_preflight_claude_binary()` (deleted in v1 as dead code) — missing `claude` binary -> exit 127 NOT_FOUND with an install hint. Deliberately claude-CLI-only; that is the feature (subscription-cheap refresh), documented as such.

### CLI surface

- `csindex quick` — options exactly as upstream (log flags) plus the standard root guard.
- `csindex embed [--backend --model --base-url --force --max-in-flight --no-conversations --no-summaries]`
- `csindex search QUERY [-k N] [--kind conversation|summary] [--backend --model --base-url]`
- `csindex embed-migrate` — the existing `backfill_kind` metadata backfill, unchanged.
- All four call the standard `_require_root_or_exit()` guard and accept command-level `--root` per the established dual-placement pattern.
- Chroma persist dir: `<export root>/.vector-db` (upstream default, kept for continuity with existing stores), overridable with `--persist` on all three embedding commands.

### Packaging

- `chromadb` returns as the `[embed]` optional extra (`uv add --optional embed chromadb`); the existing lazy import + "install the extra" error message pattern stays.
- `.gitignore` gains `.vector-db/`.

## Error handling summary

| Condition | Exit | Message contract |
|---|---|---|
| No backend configured | 78 | names all three backends and the three config channels |
| Missing creds for chosen backend | 78 | backend-specific (which env vars) |
| Collection/model mismatch or unstamped collection | 65 | both models named + single remedy (wipe persist dir, re-embed) |
| chromadb not installed | 78 | `pip install "claudesync-index[embed]"` / uv equivalent |
| Transient backend HTTP | retried, then per-source failure count | unchanged upstream semantics |
| `claude` binary missing (quick) | 127 NOT_FOUND | install hint |

## Testing

Same gates as phase 1: TDD, suite green, coverage >= 85%, ruff clean, provenance grep clean.

- `tests/test_embedding.py`: adapt upstream tests (CF mechanics, chunking, content-hash skip, per-source isolation); add backend-selection tests (config precedence, missing-backend exit 78) and compatibility-guard tests (mismatch and unstamped both exit 65).
- Contract-style tests for `OpenAIEmbedder`/`OllamaEmbedder` with mocked httpx (request shape, auth header presence/absence, response parsing, transient retry), mirroring `test_provider_*` conventions.
- `tests/test_cli.py` additions: `quick` help/root-guard/missing-binary tests; `embed`/`search`/`embed-migrate` wired with a fake embedder.
- Docs: README providers/config sections gain the embedding table + quick; `reindex.example.toml` gains `[embedding]`; all new command examples `--help`-verified. Provenance + no-.env + no-hard-wrap rules apply to everything copied or written.

## Risks / accepted tradeoffs

- OpenAI-compatible servers vary in batch-size limits; `_pack_batches` sizes stay conservative and configurable via `--max-in-flight` only (no new knobs until someone hits a real limit).
- The compatibility guard cannot distinguish two different backends serving the same model name (e.g. nomic via ollama vs openai-compat) — stamped backend+model both matching is required; this is stricter than necessary but predictable.
- `quick` remains claude-CLI-only by design; users without a subscription use `full` with any provider.
