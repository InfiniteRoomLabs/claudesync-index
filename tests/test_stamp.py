"""Frontmatter stamp: replace existing fields, insert new fields."""

from __future__ import annotations

from pathlib import Path

from reindex.stamp import read_conversation_model, stamp_frontmatter, stamp_many

FM_BASIC = """---
slug: foo
content_hash: WRONG
type: conversation
---

body line one
"""


def test_replaces_existing_field(tmp_path: Path):
    f = tmp_path / "i.md"
    f.write_text(FM_BASIC, encoding="utf-8")
    stamp_frontmatter(f, "content_hash", "REAL_HASH")
    out = f.read_text(encoding="utf-8")
    assert "content_hash: REAL_HASH" in out
    assert "WRONG" not in out
    assert "body line one" in out


def test_inserts_new_field(tmp_path: Path):
    f = tmp_path / "i.md"
    f.write_text(FM_BASIC, encoding="utf-8")
    stamp_frontmatter(f, "new_field", "value")
    out = f.read_text(encoding="utf-8")
    assert "new_field: value" in out
    assert out.count("---") >= 2


def test_stamps_value_with_special_chars(tmp_path: Path):
    f = tmp_path / "i.md"
    f.write_text(FM_BASIC, encoding="utf-8")
    stamp_frontmatter(f, "content_hash", "abc/def+xyz=123")
    assert "content_hash: abc/def+xyz=123" in f.read_text(encoding="utf-8")


def test_stamps_long_hex(tmp_path: Path):
    """Regression: 64-char hex was the field LLM hallucinated; stamping must be reliable."""
    f = tmp_path / "i.md"
    f.write_text(FM_BASIC, encoding="utf-8")
    real = "a" * 64
    stamp_frontmatter(f, "content_hash", real)
    assert f"content_hash: {real}" in f.read_text(encoding="utf-8")


def test_stamp_many_replaces_all(tmp_path: Path):
    f = tmp_path / "i.md"
    f.write_text(FM_BASIC, encoding="utf-8")
    stamp_many(f, {"content_hash": "H", "slug": "newslug", "model": "claude-test"})
    out = f.read_text(encoding="utf-8")
    assert "content_hash: H" in out
    assert "slug: newslug" in out
    assert "model: claude-test" in out


def test_stamp_no_frontmatter_creates_one(tmp_path: Path):
    f = tmp_path / "i.md"
    f.write_text("just a body\n", encoding="utf-8")
    stamp_frontmatter(f, "x", "y")
    out = f.read_text(encoding="utf-8")
    assert out.startswith("---\nx: y\n---\n")
    assert "just a body" in out


def test_stamp_unterminated_frontmatter_noop(tmp_path: Path):
    """If '---' open is present but never closes, leave file alone (don't corrupt)."""
    f = tmp_path / "i.md"
    f.write_text("---\nslug: foo\nbut never closes\n", encoding="utf-8")
    before = f.read_text(encoding="utf-8")
    stamp_frontmatter(f, "x", "y")
    assert f.read_text(encoding="utf-8") == before


def test_stamp_preserves_body(tmp_path: Path):
    f = tmp_path / "i.md"
    body = "## Section\nlots\nof\nlines\n## Another\nmore\n"
    f.write_text(f"---\nslug: foo\n---\n\n{body}", encoding="utf-8")
    stamp_frontmatter(f, "slug", "bar")
    assert body in f.read_text(encoding="utf-8")


def test_stamp_idempotent(tmp_path: Path):
    f = tmp_path / "i.md"
    f.write_text(FM_BASIC, encoding="utf-8")
    stamp_frontmatter(f, "content_hash", "X")
    a = f.read_text(encoding="utf-8")
    stamp_frontmatter(f, "content_hash", "X")
    assert f.read_text(encoding="utf-8") == a


README_WITH_MODEL = """# 10 Lesser-Known Vim Features

- **Conversation ID:** b708800e-4ef8-47ba-9cc5-ae658671ab0a
- **Model:** claude-sonnet-4-5-20250929
- **Created:** 2024-08-23T14:23:13.088099Z

---

Exported by ClaudeSync
"""


def test_read_conversation_model_from_readme(tmp_path: Path):
    (tmp_path / "README.md").write_text(README_WITH_MODEL, encoding="utf-8")
    assert read_conversation_model(tmp_path) == "claude-sonnet-4-5-20250929"


def test_read_conversation_model_missing_readme(tmp_path: Path):
    assert read_conversation_model(tmp_path) == "unknown"


def test_read_conversation_model_no_model_line(tmp_path: Path):
    (tmp_path / "README.md").write_text("# Title\n\nno model here\n", encoding="utf-8")
    assert read_conversation_model(tmp_path) == "unknown"
