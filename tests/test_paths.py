import pytest

from reindex import paths


def test_resolve_root_cli_flag_wins(tmp_path, monkeypatch):
    flag_root = tmp_path / "flag"
    flag_root.mkdir()
    (flag_root / "conversations").mkdir()
    env_root = tmp_path / "env"
    env_root.mkdir()
    (env_root / "projects").mkdir()
    monkeypatch.setenv("CSINDEX_ROOT", str(env_root))
    assert paths.resolve_root(flag_root) == flag_root.resolve()


def test_resolve_root_env_beats_cwd(tmp_path, monkeypatch):
    env_root = tmp_path / "env"
    env_root.mkdir()
    (env_root / "projects").mkdir()
    monkeypatch.setenv("CSINDEX_ROOT", str(env_root))
    monkeypatch.chdir(tmp_path)
    assert paths.resolve_root(None) == env_root.resolve()


def test_resolve_root_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("CSINDEX_ROOT", raising=False)
    (tmp_path / "conversations").mkdir()
    monkeypatch.chdir(tmp_path)
    assert paths.resolve_root(None) == tmp_path.resolve()


def test_resolve_root_rejects_non_export(tmp_path, monkeypatch):
    monkeypatch.delenv("CSINDEX_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)  # empty dir: neither conversations/ nor projects/
    with pytest.raises(paths.InvalidExportTree):
        paths.resolve_root(None)


def test_resolve_root_accepts_projects_only(tmp_path, monkeypatch):
    monkeypatch.delenv("CSINDEX_ROOT", raising=False)
    (tmp_path / "projects").mkdir()
    monkeypatch.chdir(tmp_path)
    assert paths.resolve_root(None) == tmp_path.resolve()


def test_no_reindex_env_names_left():
    import pathlib
    import re
    src = pathlib.Path(__file__).parent.parent / "src"
    hits = [p for p in src.rglob("*.py") if re.search(r"REINDEX_[A-Z_]+", p.read_text())]
    assert hits == []
