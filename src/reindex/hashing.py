"""
Content-addressed cache key helpers. All keys incorporate the relevant
schema's sha256 so any model/schema change forces re-summarization.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel

from reindex import models


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    return _sha256_bytes(p.read_bytes())


def schema_hash(model_cls: type[BaseModel]) -> str:
    """Hash the JSON schema produced by a Pydantic model."""
    import json
    return _sha256_bytes(json.dumps(model_cls.model_json_schema(), sort_keys=True).encode())


def leaf_hash(conv_file: Path) -> str:
    """Cache key for a leaf: sha256(conversation.md content + leaf schema)."""
    return _sha256_bytes(
        f"{_sha256_file(conv_file)}\n{schema_hash(models.LeafSummary)}\n".encode()
    )


def project_children_hash(project_dir: Path) -> str:
    """Cache key for a project. Aggregates child INDEX.md content_hashes,
    knowledge file hashes, and the project schema hash."""
    pairs: list[str] = []

    # Child conversation INDEX.md content_hashes from frontmatter.
    convo_dir = project_dir / "conversations"
    if convo_dir.is_dir():
        for child_idx in sorted(convo_dir.glob("*/INDEX.md")):
            slug, h = read_frontmatter_field(child_idx, "slug"), read_frontmatter_field(child_idx, "content_hash")
            if slug and h:
                pairs.append(f"convo:{slug}:{h}")

    # Knowledge files: sha256 of each file by name.
    kn_dir = project_dir / "knowledge"
    if kn_dir.is_dir():
        for kf in sorted(p for p in kn_dir.iterdir() if p.is_file()):
            pairs.append(f"kn:{kf.name}:{_sha256_file(kf)}")

    pairs.append(f"schema:{schema_hash(models.ProjectAggregate)}")
    pairs.sort()
    return _sha256_bytes("\n".join(pairs).encode() + b"\n")


def root_inputs_hash(export_dir: Path) -> str:
    """Cache key for root. Aggregates project children_hashes,
    standalone content_hashes, and the root schema hash."""
    pairs: list[str] = []

    proj_dir = export_dir / "projects"
    if proj_dir.is_dir():
        for proj_idx in sorted(proj_dir.glob("*/INDEX.md")):
            slug = read_frontmatter_field(proj_idx, "slug")
            h = read_frontmatter_field(proj_idx, "children_hash")
            if slug and h:
                pairs.append(f"project:{slug}:{h}")

    convo_dir = export_dir / "conversations"
    if convo_dir.is_dir():
        for c_idx in sorted(convo_dir.glob("*/INDEX.md")):
            slug = read_frontmatter_field(c_idx, "slug")
            h = read_frontmatter_field(c_idx, "content_hash")
            if slug and h:
                pairs.append(f"standalone:{slug}:{h}")

    pairs.append(f"schema:{schema_hash(models.RootAggregate)}")
    pairs.sort()
    return _sha256_bytes("\n".join(pairs).encode() + b"\n")


# ---------------------------------------------------------------------------
# Frontmatter helpers (no yq dependency)
# ---------------------------------------------------------------------------

def frontmatter_head(md_file: Path) -> str:
    """Return the file's leading `--- ... ---` frontmatter block (fences
    included), or the whole text when no closing fence exists. Used by the
    aggregate prompts, which feed child frontmatters — not bodies — to the
    model."""
    text = md_file.read_text(encoding="utf-8", errors="replace")
    end = text.find("\n---", 4)
    return text[: end + 4] if end != -1 else text


def read_frontmatter_field(md_file: Path, field: str) -> str | None:
    """Pull a single field from a markdown file's YAML frontmatter.
    No external dependency — simple line scan, sufficient for our flat key:value
    frontmatter (no nested structures)."""
    try:
        text = md_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 4)
    if end == -1:
        return None
    fm = text[4:end]
    prefix = f"{field}:"
    for line in fm.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None
