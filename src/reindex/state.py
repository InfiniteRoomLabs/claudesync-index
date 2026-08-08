"""
Batch resumability state. Persists active batch IDs + the per-task context
needed to finalize results after process restart.

State file: <export_root>/.batch-state.json
Format (version 1):
{
  "version": 1,
  "batches": [
    {
      "batch_id": "msgbatch_abc",
      "step": "leaf|project|root",
      "model": "claude-...",
      "submitted_at": "ISO-8601",
      "is_retry": false,
      "items": [
        { "custom_id": "...", "step_kwargs": {...} }   // step-specific finalize args
      ]
    }
  ]
}

Atomic writes via temp file + rename. Single-process assumed (no locking).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATE_FILE = ".batch-state.json"
STATE_VERSION = 1


@dataclass
class PersistedItem:
    custom_id: str
    # Serialized form of LeafItem/ProjectItem/RootItem (fields needed for finalize).
    step_kwargs: dict[str, Any]


@dataclass
class PersistedBatch:
    batch_id: str
    step: str
    model: str
    submitted_at: str
    is_retry: bool
    items: list[PersistedItem] = field(default_factory=list)
    # Which provider owns this batch ID. Tolerant default for state files
    # written before the field existed — Anthropic was the only batch
    # provider then.
    provider: str = "anthropic"


def _state_path(export_root: Path) -> Path:
    return export_root / STATE_FILE


def _now_iso_z() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_write(path: Path, content: str) -> None:
    """Write to temp file then rename. POSIX-atomic on same filesystem."""
    fd, tmp = tempfile.mkstemp(prefix=".batch-state-", suffix=".json.tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class BatchState:
    """Read/modify/write the on-disk batch state."""

    def __init__(self, export_root: Path):
        self.path = _state_path(export_root)

    def load(self) -> list[PersistedBatch]:
        if not self.path.is_file():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if raw.get("version") != STATE_VERSION:
            return []
        return [
            PersistedBatch(
                batch_id=b["batch_id"],
                step=b["step"],
                model=b["model"],
                submitted_at=b["submitted_at"],
                is_retry=b.get("is_retry", False),
                items=[PersistedItem(**it) for it in b.get("items", [])],
                provider=b.get("provider", "anthropic"),
            )
            for b in raw.get("batches", [])
        ]

    def _save(self, batches: list[PersistedBatch]) -> None:
        if not batches:
            self.clear()
            return
        payload = {
            "version": STATE_VERSION,
            "batches": [
                {
                    "batch_id": b.batch_id,
                    "step": b.step,
                    "model": b.model,
                    "submitted_at": b.submitted_at,
                    "is_retry": b.is_retry,
                    "provider": b.provider,
                    "items": [asdict(it) for it in b.items],
                }
                for b in batches
            ],
        }
        _atomic_write(self.path, json.dumps(payload, indent=2))

    def add(
        self,
        batch_id: str,
        step: str,
        model: str,
        items: list[PersistedItem],
        *,
        is_retry: bool = False,
        provider: str = "anthropic",
    ) -> None:
        existing = self.load()
        existing.append(PersistedBatch(
            batch_id=batch_id, step=step, model=model,
            submitted_at=_now_iso_z(), is_retry=is_retry,
            items=items, provider=provider,
        ))
        self._save(existing)

    def remove(self, batch_id: str) -> None:
        existing = self.load()
        existing = [b for b in existing if b.batch_id != batch_id]
        self._save(existing)

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def is_empty(self) -> bool:
        return not self.load()
