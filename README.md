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

## How it works

`csindex full` walks the export depth-first:

1. **Leaves** — every conversation (standalone, or nested under a project) gets its own summary.
2. **Projects** — each project's leaf summaries are aggregated into one project-level summary.
3. **Root** — every project and standalone-conversation summary rolls up into one root `INDEX.md` for the whole export.

Each level's `INDEX.md` stamps a content hash of its inputs. On the next run, anything whose inputs haven't changed is skipped — only new or edited conversations get re-summarized. Pass `--force` to ignore the cache and re-summarize everything.

If a provider call fails in a retryable way (malformed structured output, a parse failure), the runner retries once against that provider's configured `escalation` model — a stronger model for the cases the normal tier can't handle. Escalation is per-provider and configurable in `reindex.toml`. The root step never escalates — it already runs the strongest configured model, so retrying "up" would be a downgrade.

## Config

Optional `reindex.toml` at the export root overrides provider selection, model tiers, and pricing — see [`reindex.example.toml`](reindex.example.toml) for the full annotated shape. Copy it in and edit.

The model IDs and per-token prices baked into `claudesync-index` are point-in-time defaults. When Anthropic changes pricing or ships new models, override them in `reindex.toml` — no need to wait on a new release.

Environment variables:

| Variable | Purpose |
|---|---|
| `CSINDEX_ROOT` | Export tree path. Precedence: `--root` > `$CSINDEX_ROOT` > current directory. An invalid tree (no `conversations/` or `projects/` subdirectory) exits `65`. |
| `CSINDEX_PROVIDER` | Default provider, overriding `reindex.toml`'s `[reindex].provider`. |
| `CSINDEX_ROOT_MODEL` | Overrides the root-tier model for any provider, independent of `reindex.toml`'s per-provider model tiers. |
| `CSINDEX_COST_LOG` | Path to the JSONL cost log. Defaults to `<export>/.reindex-costs.jsonl`. |
| `CSINDEX_FAILURE_LOG` | Path to the JSONL failure log. Defaults to `<export>/.reindex-failures.jsonl`. |
| `CSINDEX_LOG_FILE` | Path to the JSONL run log. Defaults to `<export>/.reindex.log.jsonl`; disable with `--no-log-file`. |
| `LOG_FORMAT` | `human` (default on a TTY) or `json` (default otherwise) for stderr log rendering. |

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
