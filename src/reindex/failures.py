"""
Per-run failure log. Append-only JSONL. Workers and batch write here when an
item fails permanently (after retries exhausted). CLI counts at exit to
decide success-vs-tempfail return code for cron.

Lines are <4KB so POSIX O_APPEND is atomic across coroutines.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

_LOCK = threading.Lock()


def log_path() -> Path | None:
    p = os.environ.get("CSINDEX_FAILURE_LOG")
    return Path(p) if p else None


def truncate() -> Path | None:
    p = log_path()
    if p:
        p.write_text("", encoding="utf-8")
    return p


def record(
    *,
    step: str,
    slug: str,
    kind: str,
    detail: str = "",
    context: dict | None = None,
) -> None:
    """Append a failure record to the per-run failure log.

    `detail` keeps its 500-char cap for back-compat (callers that just
    pass `str(exc)`). `context` is an optional dict for structured
    per-failure data -- subprocess exit code, captured stderr/stdout,
    etc. Each string value in context is independently capped at 4KB so
    a chatty stderr doesn't bloat the line beyond a usable size.
    """
    p = log_path()
    if not p:
        return
    record_dict: dict[str, object] = {
        "step": step,
        "slug": slug,
        "kind": kind,
        "detail": detail[:500],
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
    }
    if context:
        capped: dict[str, object] = {}
        for k, v in context.items():
            capped[k] = v[:4096] if isinstance(v, str) else v
        record_dict["context"] = capped
    line = json.dumps(record_dict)
    with _LOCK, p.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def count() -> int:
    p = log_path()
    if not p or not p.exists():
        return 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            n += 1
    return n
