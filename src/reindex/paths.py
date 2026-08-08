"""Runtime export-root resolution.

Single source of truth for EXPORT_ROOT. Read it via ``paths.EXPORT_ROOT``
(attribute access), never ``from reindex.paths import EXPORT_ROOT`` — the CLI
callback reassigns the attribute at startup.
"""

from __future__ import annotations

import os
from pathlib import Path


class InvalidExportTree(Exception):
    def __init__(self, root: Path):
        self.root = root
        super().__init__(
            f"{root} is not a claudesync export tree (expected a 'conversations/' "
            "or 'projects/' directory). Point --root or $CSINDEX_ROOT at an export."
        )


def resolve_root(cli_root: Path | None) -> Path:
    """Precedence: --root > $CSINDEX_ROOT > CWD. Validates the tree."""
    env = os.environ.get("CSINDEX_ROOT")
    root = (cli_root or (Path(env) if env else Path.cwd())).resolve()
    if not ((root / "conversations").is_dir() or (root / "projects").is_dir()):
        raise InvalidExportTree(root)
    return root


def set_export_root(root: Path) -> None:
    global EXPORT_ROOT
    EXPORT_ROOT = root


# Unvalidated default so imports and --help never fail; commands must call
# require_export_root() before touching the tree.
EXPORT_ROOT = Path.cwd().resolve()
_requested_cli_root: Path | None = None


def set_requested_root(root: Path | None) -> None:
    global _requested_cli_root
    _requested_cli_root = root


def require_export_root() -> Path:
    """Validate and pin EXPORT_ROOT. Call at the top of every command."""
    root = resolve_root(_requested_cli_root)
    set_export_root(root)
    return root
