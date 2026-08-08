"""
Recompute and stamp cache-key hashes in existing INDEX.md files without
calling any LLM. Use when schema changes invalidate caches, and you want
hashes to reflect the current state without re-summarizing.

Also backfills the leaf `conversation_model` frontmatter field from each
conversation's sibling README.md. That field lives outside the leaf-schema
hash, so adding it never busts `content_hash` -- without this backfill pass,
INDEX.md files generated before the field existed would stay missing it.

WARNING: this only updates hash *fields* (+ conversation_model). Body content
stays stale relative to the schema. To get fresh content, run
`csindex full --force`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from reindex import exit_codes, hashing, log, paths, stamp
from reindex.lockfile import single_instance

app = typer.Typer(no_args_is_help=False, add_completion=False)

def collect_indexed_conversation_dirs(conversations_dir: Path) -> list[Path]:
    """Return conversation dirs containing both INDEX.md and conversation.md."""
    if not conversations_dir.is_dir():
        return []

    return [
        d
        for d in sorted(p for p in conversations_dir.iterdir() if p.is_dir())
        if (d / "INDEX.md").is_file() and (d / "conversation.md").is_file()
    ]


@app.command()
def repair_hashes(
    root: Annotated[Path | None, typer.Option(
        "--root", help="Claudesync export tree (default: $CSINDEX_ROOT, then CWD). "
        "Also accepted before the subcommand: `csindex --root PATH repair-hashes ...`.",
    )] = None,
    dry_run: bool = typer.Option(False, "--dry-run"),
    log_level: str = typer.Option(None, "--log-level"),
    log_format: str = typer.Option(None, "--log-format"),
) -> None:
    """Recompute content_hash, children_hash, inputs_hash for existing INDEX.md files."""
    # `--root` is accepted both before the subcommand (parsed by the cli app
    # callback into paths._requested_cli_root, when mounted under `csindex`)
    # and here, after `repair-hashes`; a value given here wins since it's the
    # more specific placement (mirrors `full`'s pattern).
    if root is not None:
        paths.set_requested_root(root)
    from reindex.cli import _require_root_or_exit
    _require_root_or_exit()

    load_dotenv(paths.EXPORT_ROOT / ".env")
    log.configure(level=log_level, fmt=log_format)
    rl = log.get("repair")

    try:
        with single_instance(paths.EXPORT_ROOT):
            rl.info("started", dry_run=dry_run)

            leaf_dirs: list[Path] = collect_indexed_conversation_dirs(paths.EXPORT_ROOT / "conversations")

            proj_dir = paths.EXPORT_ROOT / "projects"
            if proj_dir.is_dir():
                for proj in sorted(p for p in proj_dir.iterdir() if p.is_dir()):
                    leaf_dirs.extend(collect_indexed_conversation_dirs(proj / "conversations"))

            rl.info("leaves_found", count=len(leaf_dirs))
            for d in leaf_dirs:
                h = hashing.leaf_hash(d / "conversation.md")
                # Backfill conversation_model from the sibling README.md. The field was
                # added after these INDEX.md files were generated; since it lives
                # outside the leaf-schema hash it never busts content_hash, so existing
                # files would otherwise stay missing it forever.
                conv_model = stamp.read_conversation_model(d)
                if dry_run:
                    rl.debug("would_stamp_leaf", dir=str(d), hash=h, conversation_model=conv_model)
                else:
                    stamp.stamp_frontmatter(d / "INDEX.md", "content_hash", h)
                    stamp.stamp_frontmatter(d / "INDEX.md", "conversation_model", conv_model)
            rl.info("leaves_done")

            project_dirs: list[Path] = []
            if proj_dir.is_dir():
                for proj in sorted(p for p in proj_dir.iterdir() if p.is_dir()):
                    if (proj / "INDEX.md").is_file():
                        project_dirs.append(proj)
            rl.info("projects_found", count=len(project_dirs))
            for d in project_dirs:
                h = hashing.project_children_hash(d)
                if dry_run:
                    rl.debug("would_stamp_project", dir=str(d), hash=h)
                else:
                    stamp.stamp_frontmatter(d / "INDEX.md", "children_hash", h)
            rl.info("projects_done")

            if (paths.EXPORT_ROOT / "INDEX.md").is_file():
                h = hashing.root_inputs_hash(paths.EXPORT_ROOT)
                if dry_run:
                    rl.debug("would_stamp_root", hash=h)
                else:
                    stamp.stamp_frontmatter(paths.EXPORT_ROOT / "INDEX.md", "inputs_hash", h)
                rl.info("root_done")

            rl.info("finished")
    except RuntimeError as e:
        rl.error("lock_failed", error=str(e))
        raise typer.Exit(exit_codes.TEMPFAIL) from e
    raise typer.Exit(exit_codes.OK)


if __name__ == "__main__":
    app()
