"""
Per-invocation cost log. JSONL, append-only. Atomic across parallel workers
because lines are <4KB (POSIX O_APPEND atomic).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

_LOCK = threading.Lock()


def log_path() -> Path | None:
    p = os.environ.get("CSINDEX_COST_LOG")
    return Path(p) if p else None


def truncate() -> Path | None:
    """Truncate the cost log at run start. Returns the path."""
    p = log_path()
    if p:
        p.write_text("", encoding="utf-8")
    return p


def record(
    *,
    step: str,
    slug: str,
    cost: float,
    turns: int,
    duration_ms: int,
    model: str,
    retry: bool = False,
) -> None:
    """Append one per-invocation cost record.

    retry: True when this call was the escalation attempt after a primary
        call returned a retryable BackendFailure. Lets cost analytics
        separate "Haiku worked" vs "Haiku failed and Sonnet picked it up"
        vs "Sonnet retry also failed" populations.
    """
    p = log_path()
    if not p:
        return
    line = json.dumps({
        "step": step,
        "slug": slug,
        "cost": cost,
        "turns": turns,
        "duration_ms": duration_ms,
        "model": model,
        "retry": retry,
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
    })
    with _LOCK, p.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def aggregate(step: str | None = None) -> dict[str, int | float]:
    """Sum cost log entries, optionally filtered by step."""
    p = log_path()
    if not p or not p.exists():
        return {"n": 0, "cost": 0.0, "turns": 0, "wall_ms": 0}
    n = 0
    cost = 0.0
    turns = 0
    wall_ms = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if step and r.get("step") != step:
            continue
        n += 1
        cost += float(r.get("cost", 0))
        turns += int(r.get("turns", 0))
        wall_ms += int(r.get("duration_ms", 0))
    return {"n": n, "cost": cost, "turns": turns, "wall_ms": wall_ms}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return USD cost for a single call given token counts.

    Delegating shim: pricing now lives in config.ProviderConfig (per
    provider). This default routes through the Anthropic table for the
    remaining direct callers (batch.py) until they hold a provider.
    """
    from reindex import config

    global _default_config
    if _default_config is None:
        _default_config = config.ProviderConfig(
            name="anthropic",
            models=config._ANTHROPIC_TIERS,
            pricing=dict(config._ANTHROPIC_PRICING),
        )
    return _default_config.compute_cost(model, input_tokens, output_tokens)


_default_config = None
