"""Repair tool: recompute and stamp cache-key hashes without LLM."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from reindex import exit_codes, hashing, repair

runner = CliRunner()


def _write_index(p: Path, **fm) -> None:
    fm_lines = "\n".join(f"{k}: {v}" for k, v in fm.items())
    p.write_text(f"---\n{fm_lines}\n---\n\nbody\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_repair_finds_standalone_leaves(tmp_export, monkeypatch):
    monkeypatch.setattr(repair, "load_dotenv", lambda *a, **k: None)
    d = tmp_export / "conversations" / "foo"
    d.mkdir()
    (d / "conversation.md").write_text("content", encoding="utf-8")
    _write_index(d / "INDEX.md", slug="foo", content_hash="STAMPED_AFTER")

    res = runner.invoke(repair.app, [])
    assert res.exit_code == exit_codes.OK
    new = (d / "INDEX.md").read_text(encoding="utf-8")
    expected = hashing.leaf_hash(d / "conversation.md")
    assert f"content_hash: {expected}" in new
    assert "STAMPED_AFTER" not in new


def test_repair_backfills_conversation_model_from_readme(tmp_export, monkeypatch):
    monkeypatch.setattr(repair, "load_dotenv", lambda *a, **k: None)
    d = tmp_export / "conversations" / "foo"
    d.mkdir()
    (d / "conversation.md").write_text("content", encoding="utf-8")
    (d / "README.md").write_text(
        "# Foo\n\n- **Model:** claude-opus-4-7\n", encoding="utf-8"
    )
    # Existing INDEX.md predates the conversation_model field entirely.
    _write_index(d / "INDEX.md", slug="foo", content_hash="STAMPED_AFTER")

    res = runner.invoke(repair.app, [])
    assert res.exit_code == exit_codes.OK
    new = (d / "INDEX.md").read_text(encoding="utf-8")
    assert "conversation_model: claude-opus-4-7" in new


def test_repair_backfills_unknown_without_readme(tmp_export, monkeypatch):
    monkeypatch.setattr(repair, "load_dotenv", lambda *a, **k: None)
    d = tmp_export / "conversations" / "foo"
    d.mkdir()
    (d / "conversation.md").write_text("content", encoding="utf-8")
    _write_index(d / "INDEX.md", slug="foo", content_hash="STAMPED_AFTER")

    res = runner.invoke(repair.app, [])
    assert res.exit_code == exit_codes.OK
    assert "conversation_model: unknown" in (d / "INDEX.md").read_text(encoding="utf-8")


def test_repair_finds_project_nested_leaves(tmp_export, monkeypatch):
    monkeypatch.setattr(repair, "load_dotenv", lambda *a, **k: None)
    p = tmp_export / "projects" / "p1"
    (p / "conversations" / "c1").mkdir(parents=True)
    (p / "conversations" / "c1" / "conversation.md").write_text("hi", encoding="utf-8")
    _write_index(p / "conversations" / "c1" / "INDEX.md", slug="c1", content_hash="STAMPED_AFTER")

    res = runner.invoke(repair.app, [])
    assert res.exit_code == exit_codes.OK
    expected = hashing.leaf_hash(p / "conversations" / "c1" / "conversation.md")
    assert f"content_hash: {expected}" in (p / "conversations" / "c1" / "INDEX.md").read_text(encoding="utf-8")


def test_repair_stamps_project_index(tmp_export, monkeypatch):
    monkeypatch.setattr(repair, "load_dotenv", lambda *a, **k: None)
    p = tmp_export / "projects" / "p1"
    p.mkdir(parents=True)
    _write_index(p / "INDEX.md", slug="p1", children_hash="STAMPED_AFTER")

    res = runner.invoke(repair.app, [])
    assert res.exit_code == exit_codes.OK
    expected = hashing.project_children_hash(p)
    assert f"children_hash: {expected}" in (p / "INDEX.md").read_text(encoding="utf-8")


def test_repair_stamps_root_index(tmp_export, monkeypatch):
    monkeypatch.setattr(repair, "load_dotenv", lambda *a, **k: None)
    _write_index(tmp_export / "INDEX.md", inputs_hash="STAMPED_AFTER")

    res = runner.invoke(repair.app, [])
    assert res.exit_code == exit_codes.OK
    expected = hashing.root_inputs_hash(tmp_export)
    assert f"inputs_hash: {expected}" in (tmp_export / "INDEX.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def test_repair_dry_run_does_not_modify(tmp_export, monkeypatch):
    monkeypatch.setattr(repair, "load_dotenv", lambda *a, **k: None)
    d = tmp_export / "conversations" / "foo"
    d.mkdir()
    (d / "conversation.md").write_text("content", encoding="utf-8")
    _write_index(d / "INDEX.md", slug="foo", content_hash="ORIGINAL_VALUE")
    before = (d / "INDEX.md").read_text(encoding="utf-8")

    res = runner.invoke(repair.app, ["--dry-run"])
    assert res.exit_code == exit_codes.OK
    assert (d / "INDEX.md").read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# Edge: no INDEX.md / nothing to repair
# ---------------------------------------------------------------------------

def test_repair_empty_export(tmp_export, monkeypatch):
    monkeypatch.setattr(repair, "load_dotenv", lambda *a, **k: None)
    res = runner.invoke(repair.app, [])
    assert res.exit_code == exit_codes.OK


def test_repair_skips_dirs_without_conversation_md(tmp_export, monkeypatch):
    """Standalone dir with INDEX.md but no conversation.md → skipped (avoids stamping wrong hash)."""
    monkeypatch.setattr(repair, "load_dotenv", lambda *a, **k: None)
    d = tmp_export / "conversations" / "broken"
    d.mkdir()
    _write_index(d / "INDEX.md", slug="broken", content_hash="ORIGINAL")
    # No conversation.md present.
    res = runner.invoke(repair.app, [])
    assert res.exit_code == exit_codes.OK
    # Untouched.
    assert "content_hash: ORIGINAL" in (d / "INDEX.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Lockfile contention
# ---------------------------------------------------------------------------

def test_repair_lock_contention_returns_tempfail(tmp_export, monkeypatch):
    monkeypatch.setattr(repair, "load_dotenv", lambda *a, **k: None)
    from reindex.lockfile import single_instance
    with single_instance(tmp_export):
        res = runner.invoke(repair.app, [])
    assert res.exit_code == exit_codes.TEMPFAIL


# ---------------------------------------------------------------------------
# Repaired hash matches what reindex would compute
# ---------------------------------------------------------------------------

def test_repair_hash_matches_reindex_compute(tmp_export, monkeypatch):
    """Regression check: stamped hash should be identical to what summarize_leaf would compute."""
    monkeypatch.setattr(repair, "load_dotenv", lambda *a, **k: None)
    d = tmp_export / "conversations" / "foo"
    d.mkdir()
    (d / "conversation.md").write_text("a known sample", encoding="utf-8")
    _write_index(d / "INDEX.md", slug="foo", content_hash="WRONG")
    runner.invoke(repair.app, [])

    # If we now ran prepare_leaf, it should be a cache hit.
    from reindex.workers import prepare_leaf
    item = prepare_leaf(d)
    assert item is None  # cache hit


# ---------------------------------------------------------------------------
# Mounted as a csindex subcommand
# ---------------------------------------------------------------------------

def test_repair_hashes_is_a_csindex_subcommand():
    from typer.testing import CliRunner

    from reindex.cli import app
    result = CliRunner().invoke(app, ["repair-hashes", "--help"])
    assert result.exit_code == 0
    assert "dry-run" in result.output
