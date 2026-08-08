"""Prompt template loading: packaged defaults, per-file --prompts-dir override."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

_override_dir: Path | None = None


def set_prompts_dir(d: Path | None) -> None:
    global _override_dir
    _override_dir = d


def load_prompt(name: str) -> str:
    if _override_dir is not None:
        candidate = _override_dir / f"{name}.md"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return (resources.files("reindex") / "prompts" / f"{name}.md").read_text(encoding="utf-8")
