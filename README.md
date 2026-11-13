# claudesync-index

`csindex` builds a hierarchical, AI-generated `INDEX.md` tree over a [claudesync](https://github.com/InfiniteRoomLabs/claudesync) export of your claude.ai conversations — one summary per conversation, rolled up into per-project summaries, rolled up into a single root summary of the whole export.

It caches by content hash, so a re-run only pays to summarize conversations that are new or changed since the last pass — everything else is skipped.

> **Unofficial tool.** claudesync-index is a community project, not affiliated with or endorsed by Anthropic. It processes exports produced by [claudesync](https://github.com/InfiniteRoomLabs/claudesync) — an unofficial tool that reads your claude.ai session cookie to access the undocumented web API, which may violate Anthropic's Terms of Service and could put your account at risk. Use both tools at your own risk.

## Install

```sh
uv tool install claudesync-index
```

```sh
# one-off, no install
uvx claudesync-index --help
```

```sh
pipx install claudesync-index
```

## Quickstart

Export your conversations with claudesync, then index them:

```sh
claudesync export-all --output ~/claude-export
csindex full --root ~/claude-export --provider anthropic --wait
```

The `anthropic` provider submits work as message batches: without `--wait` the command submits and exits immediately, and a later re-run resumes and collects whatever's ready (handy for cron); `--wait` blocks until the batches finish.

Open `~/claude-export/INDEX.md` when it's done.

## Providers

| Provider | Cost | Setup |
|---|---|---|
| `anthropic` | Anthropic API, key-billed per token. Message-batch runs (the default for this provider) get 50% off list price. | `ANTHROPIC_API_KEY` env var. |
| `claude-cli` | No separate charge — consumes your Claude Pro/Max subscription quota via `claude -p`. | An authenticated `claude` binary on `PATH`. |
| `ollama` | Free, runs locally. | A reachable Ollama server (default `http://127.0.0.1:11434`, overridable in `reindex.toml`). |
| `opencode` | Whatever your opencode setup costs (Groq, Gemini, etc. — bring your own). | An authenticated `opencode` binary on `PATH`. |

`claude-cli` is the default provider. Select another with `--provider`, `--api` (alias for `--provider anthropic`), `--subscription` (alias for `--provider claude-cli`), or `$CSINDEX_PROVIDER`.

Transcripts are untrusted text, so the summarizer never gets a tool that can act on anything. `claude-cli` runs `claude -p --restricted --strict-mcp-config --tools ""` from an empty temporary directory, so no MCP server loads and the session runs in the default permission mode rather than yours. For a transcript over 200KB it allows only `--tools Read`, on a private copy of that one `conversation.md`. This needs a Claude Code release that accepts those flags; an older one fails with a message asking you to upgrade.

## How it works

`csindex full` walks the export depth-first:

1. **Leaves** — every conversation (standalone, or nested under a project) gets its own summary.
2. **Projects** — each project's leaf summaries are aggregated into one project-level summary.
3. **Root** — every project and standalone-conversation summary rolls up into one root `INDEX.md` for the whole export.

Each level's `INDEX.md` stamps a content hash of its inputs. On the next run, anything whose inputs haven't changed is skipped — only new or edited conversations get re-summarized. Pass `--force` to ignore the cache and re-summarize everything.

If a provider call fails in a retryable way (malformed structured output, a parse failure), the runner retries once against that provider's configured `escalation` model — a stronger model for the cases the normal tier can't handle. Escalation is per-provider and configurable in `reindex.toml`. The root step never escalates — it already runs the strongest configured model, so retrying "up" would be a downgrade.

## The Duet tree

Newer claudesync exports can carry a `duet/` directory beside `conversations/` and `projects/`, holding Claude Duet chat and tasks:

```
<export root>/duet/
  duet.json, tasks.json
  knowledge/TASKS.md
  conversations/<YYYY-MM-DD>-<channel-slug>/conversation.md
```

`csindex` indexes it by default, reusing the same three steps:

- Each channel-day under `duet/conversations/` is an ordinary leaf and gets its own `INDEX.md`.
- `duet/` is treated as one more project (slug `duet`), so `duet/INDEX.md` aggregates the day summaries and lists `knowledge/TASKS.md`. It is skipped while there are no day directories.
- The root `INDEX.md` includes `duet/INDEX.md` as a `=== duet ===` section.
- `csindex embed` embeds the Duet day conversations and summaries under the usual `--no-conversations` / `--no-summaries` switches.
- `csindex repair-hashes` stamps the Duet hashes too.

**What leaves your machine.** With Duet indexing on, Duet chat is sent to the configured LLM provider, and with `csindex embed` also to the embedding backend. Duet chat can quote email and calendar text written by other people. When Duet runs only because it is on by default, with no setting of yours, `csindex` logs one info line, `duet_default_on`, saying so.

**Only tool-free providers see Duet.** Quoted email is untrusted text that can carry instructions, so Duet leaves only ever reach a model that has no tools. `claude-cli`, `anthropic` and `ollama` qualify. `opencode` qualifies only when its tool-free `quick` agent is configured; without it, `csindex full` skips everything under `duet/` for that run and logs an error-level `duet_skipped_provider_not_tool_safe` line.

A root that contains only `duet/` is a valid export tree.

Exports with no `duet/` directory behave exactly as before, and the root cache key is unchanged, so nothing is re-summarized.

To turn it off, use any one of these (highest precedence first). With it off, nothing under `duet/` is read, written, hashed or embedded:

| Mechanism | Setting |
|---|---|
| CLI, on `full`, `embed` and `repair-hashes` | `--no-duet` (`--duet` forces it on) |
| Environment, including `<export>/.env` | `CSINDEX_DUET=off`. Accepted values are `1`, `true`, `yes`, `on` and `0`, `false`, `no`, `off`, in any case; any other value exits with an error. |
| `reindex.toml` | `[reindex.sources]` with `duet = false`. An unknown key in that table is an error. |

`search` and `batches resume` follow the environment and `reindex.toml`. `quick` never reads `duet/`, takes no Duet flag, and its prompt forbids changes inside `duet/`.

Turning Duet off stops new work, not past work:

- Text already sent to a provider or an embedding backend cannot be recalled.
- A pending batch submitted while Duet was on is not finalized into `duet/` once Duet is off. Its Duet items are dropped with a `duet_item_skipped` warning.
- Chunks already in the vector store stay there. They carry the metadata `tree: duet`, and `search` leaves them out while Duet is off. To remove them, delete `<export>/.vector-db/`, or point `--persist` at a new directory, and run `csindex embed --no-duet` again.

Known limitation: a claude.ai project whose directory is literally `projects/duet` would collide with this tree's project slug `duet`, and two project items with the same batch custom id would lose one result. When `projects/duet` exists, the Duet project step is skipped with a `duet_project_skipped` warning: the Duet days are still summarized, but `duet/INDEX.md` is not regenerated.

## Commands

| Command | Purpose |
|---|---|
| `csindex full` | The main pipeline — leaves -> projects -> root, as described above. |
| `csindex quick` | A cheap README/METADATA.json-only refresh via the `claude` CLI. Claude Code CLI subscribers only. |
| `csindex embed` | Embed conversation.md/INDEX.md contents into a local vector DB. Requires the `[embed]` extra. |
| `csindex search` | Semantic search over the embedded corpus. Requires the `[embed]` extra. |
| `csindex embed-migrate` | One-shot metadata backfill for collections embedded before the `kind` tag existed. Requires the `[embed]` extra. |
| `csindex repair-hashes` | Repair cache-key hashes after schema changes. |
| `csindex batches` | Inspect and manage pending Message Batches state (`list`/`show`/`cancel`/`purge`/`resume`). |

`quick` shells out to `claude -p` directly — it needs an authenticated `claude` binary on `PATH` and consumes your Claude Pro/Max subscription quota, not a separate API charge. It reads and writes only `README.md` and `METADATA.json` at the export root. The prompt it sends is packaged as `quick.md`; override it like any other template with `--prompts-dir`.

## Semantic search (optional)

`embed`/`search`/`embed-migrate` need the `[embed]` extra (`chromadb`):

```sh
uv tool install 'claudesync-index[embed]'
# or: pip install 'claudesync-index[embed]'
```

Both `embed` and `search` need an embedding backend — there's no default, so an unconfigured run exits `78` (`CONFIG`). Configure one via `--backend`, `$CSINDEX_EMBED_BACKEND`, or `[embedding].backend` in `reindex.toml` (see [`reindex.example.toml`](reindex.example.toml)):

| Backend | Setup | Default model |
|---|---|---|
| `cloudflare` | `CF_ACCOUNT_ID` + `CF_API_TOKEN` env vars (Workers AI). | `@cf/baai/bge-m3` |
| `openai` | Any OpenAI-compatible `/v1/embeddings` server. `CSINDEX_EMBED_API_KEY`, falling back to `OPENAI_API_KEY`; optional for local servers that don't check auth. | `text-embedding-3-small` |
| `ollama` | Native `/api/embed` endpoint, no auth. Defaults to `http://127.0.0.1:11434`. | `nomic-embed-text` |

`--base-url` / `$CSINDEX_EMBED_BASE_URL` / `[embedding].base_url` override the endpoint for `openai`/`ollama` (cloudflare derives its endpoint from the account id). `--model` / `$CSINDEX_EMBED_MODEL` / `[embedding].model` override the model.

Vectors persist to `<export root>/.vector-db` by default (`--persist` to override). A collection is stamped with the backend and model it was embedded with: switching backends or models means wiping `.vector-db/` and re-embedding — mixing vector spaces in one collection would corrupt search silently, so `csindex` fails loudly (exit `65`) instead.

## Config

Optional `reindex.toml` at the export root overrides provider selection, model tiers, and pricing — see [`reindex.example.toml`](reindex.example.toml) for the full annotated shape. Copy it in and edit.

The model IDs and per-token prices baked into `claudesync-index` are point-in-time defaults. When Anthropic changes pricing or ships new models, override them in `reindex.toml` — no need to wait on a new release.

Environment variables:

| Variable | Purpose |
|---|---|
| `CSINDEX_ROOT` | Export tree path. Precedence: `--root` > `$CSINDEX_ROOT` > current directory. An invalid tree (none of `conversations/`, `projects/` or `duet/` present) exits `65`. |
| `CSINDEX_DUET` | `0`, `false`, `no` or `off` skips the `duet/` tree; `1`, `true`, `yes` or `on` indexes it; any other value exits with an error. Overrides `reindex.toml`; `--duet`/`--no-duet` overrides this. See [The Duet tree](#the-duet-tree). |
| `CSINDEX_PROVIDER` | Default provider, overriding `reindex.toml`'s `[reindex].provider`. |
| `CSINDEX_ROOT_MODEL` | Overrides the root-tier model for any provider, independent of `reindex.toml`'s per-provider model tiers. |
| `CSINDEX_COST_LOG` | Path to the JSONL cost log. Defaults to `<export>/.reindex-costs.jsonl`. |
| `CSINDEX_FAILURE_LOG` | Path to the JSONL failure log. Defaults to `<export>/.reindex-failures.jsonl`. |
| `CSINDEX_LOG_FILE` | Path to the JSONL run log. Defaults to `<export>/.reindex.log.jsonl`; disable with `--no-log-file`. |
| `LOG_FORMAT` | `human` (default on a TTY) or `json` (default otherwise) for stderr log rendering. |
| `CSINDEX_EMBED_BACKEND` | Embedding backend for `embed`/`search` (`cloudflare`, `openai`, or `ollama`) — no default, see [Semantic search](#semantic-search-optional). |
| `CSINDEX_EMBED_MODEL` | Embedding model override. Defaults to a backend-specific model when omitted. |
| `CSINDEX_EMBED_BASE_URL` | Base URL override for the `openai`/`ollama` embedding backends. |

`--prompts-dir PATH` overrides individual prompt templates with your own — any template name not present in the directory falls back to the packaged default.

## Docker

```sh
docker run --rm \
  -e ANTHROPIC_API_KEY \
  -v "$HOME/claude-export:/export" \
  --user "$(id -u)" \
  ghcr.io/infiniteroomlabs/claudesync-index \
  full --root /export --provider anthropic
```

See [`docs/docker.md`](docs/docker.md) for the provider-support matrix (`claude-cli`/`ollama` need extra wiring), UID guidance, and a cron example.

## Exit codes

`csindex` follows the sysexits.h convention, cron-friendly (cron mails stderr on any non-zero exit):

| Code | Name | Meaning |
|---|---|---|
| 0 | `OK` | Success. |
| 64 | `USAGE` | Bad arguments — fix the invocation, don't retry. |
| 65 | `DATAERR` | Invalid input data (e.g. `--root` doesn't point at an export tree) — fix the data, don't retry. |
| 69 | `UNAVAILABLE` | Remote service unavailable — safe to retry. |
| 70 | `SOFTWARE` | Internal/unhandled error — retry, then alert if it recurs. |
| 74 | `IOERR` | Local I/O error — safe to retry. |
| 75 | `TEMPFAIL` | Already running, or a partial failure occurred — safe to retry. |
| 78 | `CONFIG` | Missing/invalid config (e.g. no `ANTHROPIC_API_KEY`) — fix config, don't retry. |
| 127 | `NOT_FOUND` | A required tool is missing from `PATH` (e.g. `claude`, `opencode`) — fix the install. |

## Development

```sh
uv sync
mise run test
mise run lint
mise run build
```

For a live run against the `anthropic` provider, export your key first (or use whatever secret manager you prefer):

```sh
export ANTHROPIC_API_KEY=...
uv run csindex full --root <export> --provider anthropic
```

See [`docs/adding-a-provider.md`](docs/adding-a-provider.md) to add a new provider.

## License

MIT — see [LICENSE](LICENSE).
