"""Hashing: cache keys + frontmatter parsing."""

from __future__ import annotations

from pathlib import Path

from reindex import hashing, models


def _write_index(path: Path, frontmatter: dict, body: str = "body") -> Path:
    fm_lines = "\n".join(f"{k}: {v}" for k, v in frontmatter.items())
    path.write_text(f"---\n{fm_lines}\n---\n\n{body}\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# leaf_hash
# ---------------------------------------------------------------------------

def test_leaf_hash_changes_with_content(tmp_path: Path):
    f = tmp_path / "conv.md"
    f.write_text("hello", encoding="utf-8")
    h1 = hashing.leaf_hash(f)
    f.write_text("hello world", encoding="utf-8")
    h2 = hashing.leaf_hash(f)
    assert h1 != h2


def test_leaf_hash_stable_for_same_content(tmp_path: Path):
    f = tmp_path / "conv.md"
    f.write_text("identical", encoding="utf-8")
    assert hashing.leaf_hash(f) == hashing.leaf_hash(f)


def test_leaf_hash_changes_when_schema_changes(tmp_path: Path, monkeypatch):
    """Schema-aware cache invalidation: same content + different schema -> different hash."""
    f = tmp_path / "conv.md"
    f.write_text("constant", encoding="utf-8")
    h1 = hashing.leaf_hash(f)

    monkeypatch.setattr(
        models.LeafSummary,
        "model_json_schema",
        classmethod(lambda _cls: {"changed": True}),
    )
    h2 = hashing.leaf_hash(f)
    assert h1 != h2


# ---------------------------------------------------------------------------
# children_hash + inputs_hash
# ---------------------------------------------------------------------------

def test_children_hash_empty_project(tmp_path: Path):
    proj = tmp_path / "p"
    proj.mkdir()
    h = hashing.project_children_hash(proj)
    assert isinstance(h, str)
    assert len(h) == 64


def test_children_hash_includes_child_content_hashes(tmp_path: Path):
    proj = tmp_path / "p"
    (proj / "conversations" / "foo").mkdir(parents=True)
    _write_index(
        proj / "conversations" / "foo" / "INDEX.md",
        {"slug": "foo", "content_hash": "abc123"},
    )
    h_with_one = hashing.project_children_hash(proj)

    _write_index(
        proj / "conversations" / "foo" / "INDEX.md",
        {"slug": "foo", "content_hash": "different"},
    )
    h_changed = hashing.project_children_hash(proj)
    assert h_with_one != h_changed


def test_children_hash_includes_knowledge_files(tmp_path: Path):
    proj = tmp_path / "p"
    (proj / "knowledge").mkdir(parents=True)
    (proj / "knowledge" / "spec.pdf").write_bytes(b"binary content")
    h1 = hashing.project_children_hash(proj)

    (proj / "knowledge" / "spec.pdf").write_bytes(b"changed content")
    h2 = hashing.project_children_hash(proj)
    assert h1 != h2


def test_inputs_hash_changes_when_project_changes(tmp_path: Path):
    (tmp_path / "projects" / "p").mkdir(parents=True)
    _write_index(
        tmp_path / "projects" / "p" / "INDEX.md",
        {"slug": "p", "children_hash": "h1"},
    )
    h1 = hashing.root_inputs_hash(tmp_path)
    _write_index(
        tmp_path / "projects" / "p" / "INDEX.md",
        {"slug": "p", "children_hash": "h2"},
    )
    h2 = hashing.root_inputs_hash(tmp_path)
    assert h1 != h2


def test_inputs_hash_changes_when_standalone_added(tmp_path: Path):
    (tmp_path / "conversations").mkdir()
    h_empty = hashing.root_inputs_hash(tmp_path)

    (tmp_path / "conversations" / "foo").mkdir()
    _write_index(
        tmp_path / "conversations" / "foo" / "INDEX.md",
        {"slug": "foo", "content_hash": "abc"},
    )
    h_one = hashing.root_inputs_hash(tmp_path)
    assert h_empty != h_one


# ---------------------------------------------------------------------------
# read_frontmatter_field
# ---------------------------------------------------------------------------

def test_read_frontmatter_basic(tmp_path: Path):
    f = _write_index(tmp_path / "i.md", {"slug": "foo", "content_hash": "ABC"})
    assert hashing.read_frontmatter_field(f, "slug") == "foo"
    assert hashing.read_frontmatter_field(f, "content_hash") == "ABC"


def test_read_frontmatter_missing_field(tmp_path: Path):
    f = _write_index(tmp_path / "i.md", {"slug": "foo"})
    assert hashing.read_frontmatter_field(f, "nonexistent") is None


def test_read_frontmatter_no_frontmatter(tmp_path: Path):
    f = tmp_path / "i.md"
    f.write_text("just body, no frontmatter", encoding="utf-8")
    assert hashing.read_frontmatter_field(f, "slug") is None


def test_read_frontmatter_unterminated(tmp_path: Path):
    f = tmp_path / "i.md"
    f.write_text("---\nslug: foo\nbody never closed\n", encoding="utf-8")
    assert hashing.read_frontmatter_field(f, "slug") is None


def test_read_frontmatter_missing_file(tmp_path: Path):
    assert hashing.read_frontmatter_field(tmp_path / "doesnt_exist.md", "slug") is None


def test_read_frontmatter_value_with_spaces(tmp_path: Path):
    f = _write_index(tmp_path / "i.md", {"title": "Multi Word Title"})
    assert hashing.read_frontmatter_field(f, "title") == "Multi Word Title"
