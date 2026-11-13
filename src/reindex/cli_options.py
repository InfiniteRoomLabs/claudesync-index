"""Typer option types shared by `cli` and `repair`.

They live here, not in cli.py, because cli imports repair to mount
`repair-hashes`; repair importing cli at module level would be a cycle.
"""

from __future__ import annotations

from typing import Annotated

import typer

# The default is None ("not given") so $CSINDEX_DUET and reindex.toml still apply.
DuetOption = Annotated[bool | None, typer.Option(
    "--duet/--no-duet",
    help="Index the duet/ tree. Default: $CSINDEX_DUET, then reindex.toml "
    "[reindex.sources].duet, then on.",
)]
