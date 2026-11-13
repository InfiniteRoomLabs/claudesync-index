# Index the `duet/` tree -- Design

**Date:** 2026-11-13
**Status:** approved in conversation 2026-11-13.
**Upstream contract:** claudesync `docs/superpowers/specs/2026-11-13-duet-design.md`, section "Export tree contract".

> **STATE AS OF 2026-11-13** (review-gate fix pass). Changes from the design below:
>
> - **Summarizer hardening (all leaves).** `claude-cli` runs `claude -p --restricted --strict-mcp-config`
>   with `--tools ""` inline and `--tools Read` in file mode, from a fresh temporary `cwd`. File mode copies
>   only `conversation.md` into that directory, passes it as the only `--add-dir`, and rewrites the prompt
>   to name the copy. An older claude that rejects the flags is a non-retryable upgrade error, never a
>   fallback. A JSON-array envelope is accepted (last `"type": "result"` event).
> - **Tool-safe providers.** Providers declare `tool_safe` (fails closed). `full` skips all Duet work on a
>   provider that is not, logging `duet_skipped_provider_not_tool_safe` at error level. opencode is tool-safe
>   only with its `quick` agent.
> - **Flags.** `--duet/--no-duet` is on `full`, `embed` and `repair-hashes`; `quick` never reads `duet/` and
>   takes no flag. `quick.md` now protects `duet/`.
> - **`projects/duet` collision.** This repo states no claudesync rule for project directory names, so a
>   reserved slug such as `duet--tree` cannot be shown collision-free. The slug stays `duet`, and the Duet
>   project step is skipped with `duet_project_skipped` while `projects/duet` exists.
> - **Pending batches.** Duet items persist `"duet": true`; resuming with Duet off drops them
>   (`duet_item_skipped`), so nothing is finalized into `duet/` after it is turned off.
> - **Vector store.** Duet chunks carry `tree: duet`; `search` excludes them while Duet is off. Disabling
>   stops new uploads only; purging means re-embedding into a fresh store.
> - **Switch values.** `$CSINDEX_DUET` accepts only `1/true/yes/on` and `0/false/no/off`; anything else exits.
>   A non-table `[reindex.sources]` or an unknown key in it exits. Resolution happens after `<export>/.env`
>   is loaded.

## Context

claudesync gains a `duet/` tree beside `conversations/` and `projects/`, holding
Claude Duet chat and tasks. It is deliberately project-shaped so this indexer can
reuse its leaf and project steps:

```
<export root>/duet/
  duet.json
  tasks.json
  knowledge/TASKS.md
  conversations/<YYYY-MM-DD>-<channel-slug>/
    conversation.md        one channel-day of Duet chat
    messages.json, README.md, .claudesync-state.json
    INDEX.md               written by this tool
  INDEX.md                 written by this tool (project-level aggregate)
```

The owner wants Duet indexed by default, with the usual configuration
mechanisms able to turn it off.

## Decision

- **Setting `duet` (default on).** Precedence, matching provider selection:
  CLI `--duet/--no-duet` on `full` and `embed` and `repair-hashes` >
  `$CSINDEX_DUET` (`1`, `true`, `yes`, `on` enable; `0`, `false`, `no`, `off`
  disable; anything else exits) > `[reindex.sources].duet` in `reindex.toml` >
  `true`. `quick` never reads `duet/`. Resolved by one function in
  `config.py`, after `<export>/.env` is loaded. `reindex.example.toml`
  documents it.
- **Export-tree validation.** A root containing only `duet/` is a valid export
  tree. The error message names all three directories.
- **Leaves.** When enabled and `duet/conversations/` exists, its child
  directories join the nested leaf set (scope `nested` and `both`). They are
  ordinary leaves: `conversation.md` in, `INDEX.md` out, same prompts, same
  cache key.
- **Project step.** `duet/` is treated as one more project directory, slug
  `duet`, so `duet/INDEX.md` aggregates the day leaves and lists
  `knowledge/TASKS.md`. It is skipped when `duet/conversations/` has no leaves.
- **Root step.** `duet/INDEX.md`, when present and enabled, is included in the
  root payload as `=== duet ===` and in `root_inputs_hash` as `duet:<children_hash>`.
  With Duet disabled or absent the root hash and payload are byte-identical to
  today, so existing exports see no cache invalidation.
- **Embed and search.** `iter_sources` yields Duet leaves and the Duet project
  summary under the same `conversations` / `summaries` switches, gated by the
  setting.
- **Disabled means invisible.** With the setting off, no step reads, writes,
  hashes or embeds anything under `duet/`.

## Known limitation

A claude.ai project whose directory is literally `projects/duet` would share the
slug `duet`, and so the batch custom id, with the Duet project item. Batch
results are matched by custom id, so one of the two would be lost. Handled by
skipping the Duet project step with a `duet_project_skipped` warning while
`projects/duet` exists: Duet days are still summarized, `duet/INDEX.md` is not
regenerated, and the root includes it only if one exists from an earlier run.

## Verification

Tests for: precedence of the setting across CLI, env and TOML; a `duet/`-only
root validating; Duet leaves discovered and limited correctly; project step
including `duet/`; root payload and hash with and without Duet (and unchanged
when disabled or absent); embed sources; nothing touched when disabled.
`ruff check` clean and coverage at or above the CI floor of 85%.
