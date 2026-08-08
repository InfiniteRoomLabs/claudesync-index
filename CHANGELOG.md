# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Release infrastructure: `.github/workflows/release.yml` workflow for tag-triggered PyPI publishing (trusted publishers) and GHCR image builds. Includes tag/version consistency validation and multi-job dependency chain (check → test → pypi/docker). `docs/release-bootstrap.md` provides one-time setup checklist for PyPI environment, GitHub Actions environment, and organization package settings.
- Project scaffold: `pyproject.toml` (package `reindex`, console script `csindex`), `uv.lock`
  with the source repo's dependency floors, `mise.toml` pinning the `uv` CLI toolchain
  and defining `test`/`lint`/`build`/`docker-build` tasks, MIT `LICENSE`, and `.gitignore`.
- Indexer core imported from the private source repo: `src/reindex/*.py` (CLI, batch runner,
  providers for claude-cli/anthropic/ollama/opencode, hashing, models, config, logging) plus
  the matching `tests/` suite (338 tests, all green). The `quick` command, the `work` sub-app
  (`workflow_cli.py`), and `embed`/`search`/`embed-migrate` (`embedding.py`) were left behind —
  cut from the public v1 surface per the extraction plan.

- Runtime export-root selection: `csindex [--root PATH] full ...` / `$CSINDEX_ROOT` / CWD
  fallback, replacing import-time root resolution in `reindex.paths`. Precedence is
  `--root` > `$CSINDEX_ROOT` > CWD; an export tree is any directory containing a
  `conversations/` or `projects/` subdirectory. Invalid trees now exit 65 (`DATAERR`)
  instead of silently resolving to the wrong directory. `--root` is accepted both before
  the subcommand (`csindex --root PATH full`) and directly on `full` (`csindex full --root
  PATH`); `csindex full --help` and `csindex batches --help` still work outside an export
  tree since validation happens on command entry, not at `--help` time. `csindex batches
  {list,show,cancel,purge,resume}` validate the same way.
- `csindex repair-hashes [--dry-run]`: the standalone `repair.py` typer app is now mounted
  as a `csindex` subcommand instead of a separate executable, following the same
  `--root` / `$CSINDEX_ROOT` / CWD resolution as `full`. `--root` is accepted both before
  the subcommand (`csindex --root PATH repair-hashes`) and directly on `repair-hashes`
  (`csindex repair-hashes --root PATH`).
- `csindex batches {list,show,cancel,purge,resume}` now also accept `--root PATH` placed
  after the subcommand (previously only `csindex --root PATH batches ...` worked; putting
  `--root` after the subcommand exited 2 even though the validation error text told users
  to use it).
- Prompt templates (`conversation-summary`, `conversation-summary-file`, `project-aggregate`,
  `root-aggregate`) now ship as package data under `src/reindex/prompts/*.md` instead of
  being read from `<export root>/prompts/`, via the new `reindex.prompt_loader` module.
  `csindex --prompts-dir PATH` overrides individual templates (per-file fallback to the
  packaged default when a name isn't present in the override directory). The `quick`
  command's `reindex.md` prompt was left behind — it belongs to the cut `quick` mode.
- Environment variables use the `CSINDEX_*` prefix: `CSINDEX_ROOT`, `CSINDEX_PROVIDER`,
  `CSINDEX_ROOT_MODEL`, `CSINDEX_COST_LOG`, `CSINDEX_FAILURE_LOG`, `CSINDEX_LOG_FILE`.
- `Dockerfile` (multi-stage: `ghcr.io/astral-sh/uv:python3.13-trixie-slim` builder ->
  `python:3.13-slim-trixie` runtime) and `.dockerignore` (allowlist style). Runs as
  non-root `csindex` (uid 1000), bakes `CSINDEX_PROVIDER=anthropic` as the image
  default, `ENTRYPOINT ["csindex"]` with `CMD ["--help"]`. The runtime stage installs
  via `uv pip install --system -r requirements.txt --no-deps *.whl` (the frozen
  dependency set `uv export --frozen --no-dev` pulled from `uv.lock`, plus our own
  wheel with `--no-deps`) rather than plain `pip install *.whl` — this pins the image's
  installed dependency versions to the lockfile instead of letting the wheel's loose
  PEP 440 ranges re-resolve against whatever's newest on the index at build time; `uv`
  itself is removed from the final layer afterward. `mise.toml`'s `docker-build` task
  tags the result `claudesync-index:dev`.
- CI workflow (`.github/workflows/ci.yml`): test matrix over Python 3.12 and 3.13 with
  ruff linting and pytest coverage gating at 85%; smoke job validates wheel and sdist
  CLI invocations via `uvx`. Also supports `workflow_call` for reuse by the release
  workflow (Task 11).

### Removed
- `fnox.toml` and `fnox` from `mise.toml` `[tools]` — API-key injection via fnox is unused; contributors export `ANTHROPIC_API_KEY` directly.

### Changed
- Lint: repo now passes `ruff check .` cleanly (173 findings -> 0). Most were auto-fixed
  (import sorting, deprecated typing imports, unused imports) or hand-fixed where small
  and behavior-preserving (`raise ... from err` exception chaining, explicit `zip(...,
  strict=True)`, unused test locals). `line-length` was bumped from an unenforced 100 to
  120 (the codebase never actually respected 100; bumping cleared ~85 line-too-long hits
  without reflowing dozens of unrelated lines) and `N818` (exception-name `Error` suffix)
  was added to `ignore` since `ProviderFailure`/`InvalidExportTree` are public API
  referenced across 16 files — a pure-cosmetic rename with no behavioral upside.

### Fixed
- Scrubbed a specific-ops-tool reference from the packaged `conversation-summary` /
  `conversation-summary-file` prompt templates' `tech_stack` example list (swapped for
  `terraform`) — the provenance constraints ban references to the maintainer's personal
  infrastructure in any shipped file, and the initial prompt-scrub pass missed it since
  it was buried in a field-guideline example list rather than a structural path/script
  reference.
- Scrubbed remaining private-provenance references from provider modules and comments:
  the Ollama provider's module docstring, timeout comment, and preflight failure message
  no longer name a specific remote host or its access helper; `config.py`'s
  `$CSINDEX_ROOT_MODEL` docstring/comment and the escalation-tier comment no longer cite
  pre-rename history or a named incident; `claude_cli.py`'s `--disallowed-tools` comment
  describes the observed failure mode generically instead of naming the incident.
- `opencode` provider: `--agent quick` is now conditional on the agent file actually
  existing (`$XDG_CONFIG_HOME/opencode/agent/quick.md`, falling back to `~/.config`)
  instead of being hardcoded — a fresh install now runs against opencode's default agent
  with a one-line warning instead of requiring the file to be pre-configured.

### Docs
- `TODO.md`: Project backlog documenting plans to generalize `quick`/`work`/`embed` features from the private upstream repo for public use, with configurable embedding backends and zero machine-specific assumptions.
- `README.md`: pitch, install (`uv tool install` / `uvx` / `pipx`), quickstart, a provider cost/setup table, how the leaf/project/root pipeline and content-hash caching work, config (`reindex.toml` + `CSINDEX_*` env vars + `--prompts-dir`), a Docker one-liner, the exit-code table (from `exit_codes.py`), a development section (`uv sync` / `mise run test|lint|build`, plain `export ANTHROPIC_API_KEY` for live runs — no fnox, it was already dropped from this repo), and the unofficial-tool disclaimer adapted from claudesync's.
- `docs/adding-a-provider.md`: adapted from the private upstream repo's copy — the provider-selection env var renamed to `$CSINDEX_PROVIDER`, otherwise unchanged (provider extension steps didn't reference any cut features or private paths).
- `docs/docker.md`: provider-support matrix for the container (`anthropic` out of the box; `claude-cli`/`ollama`/`opencode` need mounted auth or `--network host`), `--user "$(id -u)"` UID guidance, and a cron example. Written against the image Task 9 builds next.
- `reindex.example.toml`: annotated copy of the `reindex.toml` shape documented in `config.py`'s docstring — provider selection, per-tier model overrides, pricing overrides, and `[reindex.providers.ollama] base_url`.

### Docs (fix round 1)
- `README.md`: dropped the anthropic-provider row's `.env`-at-export-root suggestion per the no-`.env`-suggestions-in-docs override (the code's own dotenv auto-load is unaffected — this only changes what the docs advertise); noted the root step never escalates (`escalate=False`, since root already runs the strongest configured model).
- `reindex.example.toml`: corrected the `escalation` comment — omitting the key keeps the built-in default escalation model, only an explicit empty string disables it (verified against `runner.py`'s `not escalation` check); corrected the pricing comment — prefixes match the *requested* model id (`req.model`), not one "returned by the API".
- `docs/docker.md`: fixed the exit-code retry guidance to match `exit_codes.py`/the README's table (`75` TEMPFAIL, not just `69`/`74`, is cron-safe to retry; `70` SOFTWARE is retry-then-alert, not a plain retry); reworked the cron example so `ANTHROPIC_API_KEY` is actually populated — cron doesn't inherit shell exports, so bare `-e ANTHROPIC_API_KEY` resolved to an empty value (exit `78`). Now set inline on the crontab line from a secret file.

### Docs (fix round 2)
- `docs/docker.md`: added a caveat to the UID guidance section — `--user "$(id -u)"` only
  resolves `$HOME` to `/home/csindex` (needed for the `claude-cli`/`opencode` mount paths
  documented above it) when the host UID happens to be `1000`, matching the image's
  built-in `csindex` user. Verified in-container: any other UID has no `/etc/passwd`
  entry, so Docker's passwd lookup falls back to `$HOME=/`, silently breaking those mount
  paths. Documented the fix (`-e HOME=/home/csindex` alongside `--user`) and verified it
  restores the expected `$HOME`.

### Public-readiness pass
- Full pre-publish audit: secret/credential scan, key-shaped-string scan, personal-provenance
  and hard-privacy-rule grep, and license/doc-accuracy checks across the working tree and all
  15 commits (content and messages). No leaks found — repo has never been pushed to any remote.
- `docs/release-bootstrap.md`: added the missing **PyPI Project Name** field to the trusted-publisher
  setup steps; corrected "Confirm both `v0.1.0` and `latest` tags are present" to `0.1.0` — GHCR
  tags strip the `v` prefix (`${GITHUB_REF_NAME#v}` in the release workflow).
- `.github/workflows/release.yml`: added `contents: read` to the `pypi` job's permissions
  (least-privilege alongside the existing `id-token: write`); added `skip-existing: true` to the
  `pypa/gh-action-pypi-publish` step so a re-run after a partial failure doesn't hard-fail on an
  already-published version.
- `docs/adding-a-provider.md` entry in this changelog: reworded to describe the env-var rename
  without spelling out the retired env-var prefix, per the constraint against naming that
  prefix anywhere in this file.
- Corrected the lint-cleanup entry's finding count (161 -> 173) to match the actual `ruff check .`
  output recorded in that session's report.
- `docs/docker.md`: "a root-only file" reworded to "a chmod-600 file readable only by your user" —
  more precise about what makes the file safe to read the API key from.
- `src/reindex/batch.py`: dropped a stale `# noqa: F401` on the `Awaitable, Callable` import —
  both names are genuinely used in `run()`'s and `resume_pending()`'s parameter annotations;
  ruff passes clean without the suppression. (A queued instruction to delete the import outright
  was incorrect — verified via Pyright/ruff and kept the import, only removing the dead noqa.)

### Final review fixes
- `README.md`/`docs/docker.md`: added `--wait` to the one-shot `anthropic`-provider quickstart
  examples, with a sentence explaining the submit-and-exit model — without `--wait` the batch
  provider submits and exits, and a re-run resumes/collects results, which is why the Docker
  cron example intentionally omits the flag.
- `src/reindex/cli.py`, `src/reindex/repair.py`, `src/reindex/batches_cli.py`,
  `src/reindex/providers/base.py`, `src/reindex/batch.py`: updated stale `reindex <subcommand>`
  command examples in docstrings/comments to `csindex <subcommand>` — the installed CLI name,
  not the `reindex` package name (which is unchanged).
- This changelog: reworded the fix-round-entry above about the retired env-var prefix so the
  entry itself no longer contains the literal it was describing.
- `.gitignore`: added `.env`, since the code auto-loads `<export>/.env` and a stray one
  shouldn't land in a commit.
