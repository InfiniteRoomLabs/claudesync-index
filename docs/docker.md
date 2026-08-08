# Running in Docker

The published image (`ghcr.io/infiniteroomlabs/claudesync-index`) runs `csindex` with no host dependency beyond Docker. It defaults to the `anthropic` provider, since that's the only one that needs no extra host wiring.

## Provider support in the container

| Provider | Out of the box | Notes |
|---|---|---|
| `anthropic` | Yes | Pass `ANTHROPIC_API_KEY` via `-e`. Image default. |
| `ollama` | Needs wiring | The container's own `localhost` is not the host's. On Linux, add `--network host` so `http://127.0.0.1:11434` reaches a host-run Ollama. Otherwise point `[reindex.providers.ollama] base_url` in `reindex.toml` at a reachable address (LAN IP, Tailscale IP, or a container on the same Docker network). |
| `claude-cli` | Needs wiring | Requires an authenticated `claude` binary and its config directory. Mount your host's `~/.claude` read-only into the container, e.g. `-v "$HOME/.claude:/home/csindex/.claude:ro"`. |
| `opencode` | Needs wiring | Same pattern as `claude-cli` — mount wherever `opencode`'s auth lives (e.g. `~/.local/share/opencode`) read-only. |

## Run

```sh
docker run --rm \
  -e ANTHROPIC_API_KEY \
  -v "$HOME/claude-export:/export" \
  --user "$(id -u)" \
  ghcr.io/infiniteroomlabs/claudesync-index \
  full --root /export --provider anthropic --wait
```

The `anthropic` provider submits work as message batches: without `--wait` this command submits and exits immediately, and a later re-run resumes and collects whatever's ready; `--wait` blocks the container until the batches finish. The cron example below deliberately omits `--wait` — see there for why that's the right call for a scheduled run.

## UID guidance

`--user "$(id -u)"` runs the container as your host UID, so the `INDEX.md` files, hash caches, and log files it writes into the mounted export tree come out owned by you instead of the image's built-in user (or root). This matters most on Linux hosts, where the bind-mounted export directory keeps your host UID/GID regardless of what user the container runs as internally.

**Caveat for `claude-cli`/`opencode` plus `--user`:** the image's built-in user is `csindex`, UID 1000, home `/home/csindex`. `--user "$(id -u)"` only works out of the box for the `claude-cli`/`opencode` mount paths above if your host UID is also `1000` — then the container's `/etc/passwd` lookup resolves `$HOME` to `/home/csindex` as expected. Any other host UID has no matching `/etc/passwd` entry, so `$HOME` inside the container falls back to `/` and the `claude`/`opencode` binaries won't find config mounted at `/home/csindex/...`. If your UID isn't 1000, add `-e HOME=/home/csindex` alongside `--user "$(id -u)"` to keep `$HOME` pointed at the mount.

## Cron example

Re-index nightly; content-hash caching means only conversations that changed since the last run get re-summarized. This example intentionally leaves off `--wait`: each night's run submits a batch and exits without blocking the cron job, and the *next* night's run picks up and finalizes whatever finished server-side in the meantime before submitting new work — cron-friendly by design.

Cron doesn't inherit your shell's exported variables, so `-e ANTHROPIC_API_KEY` alone passes through nothing — read the key from wherever you keep it (a secret manager, a chmod-600 file readable only by your user) and set it inline on the crontab line:

```cron
0 2 * * * ANTHROPIC_API_KEY=$(cat "$HOME/.config/claudesync-index/anthropic-api-key") docker run --rm -e ANTHROPIC_API_KEY -v "$HOME/claude-export:/export" --user "$(id -u)" ghcr.io/infiniteroomlabs/claudesync-index full --root /export --provider anthropic >> "$HOME/claude-export/.reindex-cron.log" 2>&1
```

Exit codes follow the table in the main [README](../README.md#exit-codes): `69` (remote unavailable), `74` (local I/O), and `75` (already running, or a partial failure) are safe to retry; `70` (internal error) is worth retrying too but should alert if it keeps recurring; `64`, `65`, `78`, and `127` mean something needs a human fix, not another cron run.
