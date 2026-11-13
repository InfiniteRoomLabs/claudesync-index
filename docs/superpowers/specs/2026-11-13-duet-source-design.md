# Index the `duet/` tree -- Design

**Date:** 2026-11-13
**Status:** approved in conversation 2026-11-13.
**Upstream contract:** claudesync `docs/superpowers/specs/2026-11-13-duet-design.md`, section "Export tree contract".

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
  CLI `--duet/--no-duet` on `quick`, `full` and `embed` > `$CSINDEX_DUET`
  (`0`, `false`, `no`, `off` disable; anything else enables) >
  `[reindex.sources].duet` in `reindex.toml` > `true`. Resolved by one function
  in `config.py`. `reindex.example.toml` documents it.
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
slug `duet` in cost logs and batch custom ids. Documented, not handled.

## Verification

Tests for: precedence of the setting across CLI, env and TOML; a `duet/`-only
root validating; Duet leaves discovered and limited correctly; project step
including `duet/`; root payload and hash with and without Duet (and unchanged
when disabled or absent); embed sources; nothing touched when disabled.
`ruff check` clean and coverage at or above the CI floor of 85%.
