# Index the `duet/` tree -- Implementation Plan

Spec: `docs/superpowers/specs/2026-11-13-duet-source-design.md` (binding).
Method: test first for every task, watch it fail, then implement. One commit per task.

## How the setting is threaded

One resolver, `config.resolve_duet(export_root, cli_value)`, mirrors
`resolve_provider_name`: CLI flag, then `$CSINDEX_DUET`, then
`[reindex.sources].duet`, then `True`.

The resolved bool is passed as an explicit `duet` keyword parameter. There is no
new module global. Provider selection already works this way (resolved in the CLI,
handed down). `paths.EXPORT_ROOT` is module state only because every command needs
the root; a boolean that changes per call does not justify the same treatment.

Low-level helpers (`hashing`, `workers`, `embedding`, CLI discovery) default to
`duet=False`. Only the CLI layer resolves the setting and passes it, so a caller
that forgets cannot accidentally read `duet/`, and non-Duet behaviour is unchanged
by construction.

## Tasks

1. **Config resolver** (`config.py`, `tests/test_config.py`).
   Precedence across CLI, env, TOML, default. Env falsy set is `0 false no off`
   (case-insensitive); an empty env var counts as unset, like the provider env var.
   A non-boolean TOML value exits with a clear message.

2. **Root validation** (`paths.py`, `tests/test_paths.py`).
   A root with only `duet/` validates. The error names `conversations/`,
   `projects/` and `duet/`.

3. **Discovery** (`cli.py`, `tests/test_cli.py`).
   `_duet_dirs`; `_nested_dirs(root, duet=)` appends Duet day directories;
   `_project_dirs(root, duet=)` appends `duet/` only when it has leaf directories;
   `_gather_leaves(scope, limit, *, duet)` threads it, `--limit` applies after.

4. **Hashing and root payload** (`hashing.py`, `workers.py`, tests).
   `root_inputs_hash(..., duet=)` adds `duet:<children_hash>`; `prepare_root(..., duet=)`
   appends `=== duet ===` last. Invariant tests: with Duet disabled, or `duet/` absent,
   the hash equals a pinned reference computed without any Duet code, and the payload
   is byte-identical. `prepare_project` and `project_children_hash` need no change:
   `duet/` already has the `conversations/` and `knowledge/` layout they read.

5. **Embed sources** (`embedding.py`, `tests/test_embedding.py`).
   `_leaf_dirs` and `iter_sources` take `duet=`; `embed_corpus` passes it on.
   The Duet summary is `duet/INDEX.md`, yielded under the existing `summaries` switch.

6. **CLI wiring** (`cli.py`, `repair.py`, tests).
   `--duet/--no-duet` on `quick`, `full`, `embed`, default "not given" (`None`).
   `full` and `embed` pass the resolved value down. `repair-hashes` resolves it from
   env and TOML (no flag) so its root hash matches what `full` stamps, and when
   enabled it also stamps the Duet day leaves and the `duet/` project hash.
   `quick` accepts and resolves the flag for a uniform surface. Its prompt never reads
   `duet/` and prompts are out of scope, so the flag has no further effect there.

7. **Disabled means invisible** (tests).
   With the setting off, a `duet/` tree containing unreadable or poisoned files is
   never opened: no INDEX.md written, nothing hashed, nothing embedded.

8. **Docs** (`reindex.example.toml`, `README.md`, `CHANGELOG.md`).
   Document `[reindex.sources] duet = true`, the tree, and how to turn it off.

## Gates before finishing

```
uv sync --frozen --all-extras
uv run ruff check .
uv run pytest tests/ --cov=reindex --cov-fail-under=85
```

## Known limits (not handled, as per spec)

A claude.ai project literally named `duet` shares the slug `duet` in cost logs and
batch custom ids.
