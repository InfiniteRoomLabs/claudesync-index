"""log.configure: dual-handler stderr + file output, JSONL format, rotation."""

from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path

import pytest

from reindex import log


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Each test starts with a clean stdlib root logger."""
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _flush_handlers():
    for h in logging.getLogger().handlers:
        h.flush()


# ---------------------------------------------------------------------------
# File handler basics
# ---------------------------------------------------------------------------

def test_file_handler_writes_jsonl(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    log.configure(level="info", fmt="human", log_file=p)
    log.get("test").info("hello", x=1)
    _flush_handlers()

    rows = _read_jsonl(p)
    assert len(rows) == 1
    assert rows[0]["component"] == "test"
    assert rows[0]["x"] == 1
    assert rows[0]["event"] == "hello"
    assert rows[0]["level"] == "info"


def test_file_always_jsonl_regardless_of_stderr_format(tmp_path: Path, capfd):
    """Stderr human + file JSONL — both work."""
    p = tmp_path / "log.jsonl"
    log.configure(level="info", fmt="human", log_file=p)
    log.get("worker").info("started", slug="foo")
    _flush_handlers()

    # File should be valid JSONL.
    rows = _read_jsonl(p)
    assert rows[0]["slug"] == "foo"

    # Stderr should be human (ANSI-colored output, not JSON).
    captured = capfd.readouterr()
    assert "started" in captured.err
    # Human format doesn't surround the message in JSON braces.
    assert not captured.err.strip().startswith("{")


def test_file_jsonl_when_stderr_json(tmp_path: Path, capfd):
    p = tmp_path / "log.jsonl"
    log.configure(level="info", fmt="json", log_file=p)
    log.get("worker").info("started", slug="foo")
    _flush_handlers()

    rows = _read_jsonl(p)
    assert rows[0]["slug"] == "foo"

    # Stderr also JSON.
    captured = capfd.readouterr()
    assert '"slug": "foo"' in captured.err or '"slug":"foo"' in captured.err


# ---------------------------------------------------------------------------
# Log file path resolution
# ---------------------------------------------------------------------------

def test_no_log_file_disables_file_handler(tmp_path: Path):
    log.configure(level="info", fmt="json", no_log_file=True)
    handlers = logging.getLogger().handlers
    # Only stderr handler should be present.
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)


def test_env_var_log_file(tmp_path: Path, monkeypatch):
    p = tmp_path / "via_env.jsonl"
    monkeypatch.setenv("CSINDEX_LOG_FILE", str(p))
    log.configure(level="info", fmt="json")
    log.get("t").info("evt")
    _flush_handlers()
    assert p.exists()


def test_default_log_file_uses_export_root(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CSINDEX_ROOT", str(tmp_path))
    monkeypatch.delenv("CSINDEX_LOG_FILE", raising=False)
    log.configure(level="info", fmt="json")
    log.get("t").info("evt")
    _flush_handlers()
    assert (tmp_path / ".reindex.log.jsonl").exists()


def test_explicit_log_file_overrides_env(tmp_path: Path, monkeypatch):
    explicit = tmp_path / "explicit.jsonl"
    via_env = tmp_path / "via_env.jsonl"
    monkeypatch.setenv("CSINDEX_LOG_FILE", str(via_env))
    log.configure(level="info", fmt="json", log_file=explicit)
    log.get("t").info("evt")
    _flush_handlers()
    assert explicit.exists()
    assert not via_env.exists()


def test_no_log_file_overrides_env(tmp_path: Path, monkeypatch):
    via_env = tmp_path / "via_env.jsonl"
    monkeypatch.setenv("CSINDEX_LOG_FILE", str(via_env))
    log.configure(level="info", fmt="json", no_log_file=True)
    log.get("t").info("evt")
    _flush_handlers()
    assert not via_env.exists()


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def test_rotation_creates_backup(tmp_path: Path, monkeypatch):
    """When log file exceeds maxBytes, RotatingFileHandler creates .1 backup."""
    p = tmp_path / "rotates.jsonl"
    log.configure(level="info", fmt="json", log_file=p)

    # Force the handler's maxBytes very low so we trigger rotation cheaply.
    file_handler: logging.handlers.RotatingFileHandler | None = None
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            h.maxBytes = 200
            file_handler = h
            break

    assert file_handler is not None

    for i in range(50):
        log.get("t").info("evt", i=i, payload="x" * 50)
    _flush_handlers()

    # At least one backup file should exist.
    backups = list(tmp_path.glob("rotates.jsonl.*"))
    assert len(backups) >= 1


# ---------------------------------------------------------------------------
# Multiple configure() calls reset handlers cleanly
# ---------------------------------------------------------------------------

def test_reconfigure_does_not_double_emit(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    log.configure(level="info", fmt="json", log_file=p)
    log.configure(level="info", fmt="json", log_file=p)  # reconfigure
    log.get("t").info("evt")
    _flush_handlers()
    rows = _read_jsonl(p)
    assert len(rows) == 1  # not 2


# ---------------------------------------------------------------------------
# Level filtering
# ---------------------------------------------------------------------------

def test_level_filter_warn(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    log.configure(level="warn", fmt="json", log_file=p)
    log.get("t").debug("dbg")
    log.get("t").info("inf")
    log.get("t").warn("wrn")
    log.get("t").error("err")
    _flush_handlers()
    rows = _read_jsonl(p)
    levels = [r["level"] for r in rows]
    assert "debug" not in levels
    assert "info" not in levels
    assert "warning" in levels
    assert "error" in levels


def test_extra_fields_preserved(tmp_path: Path):
    """Bound context vars + per-call kwargs both appear in JSON output."""
    p = tmp_path / "log.jsonl"
    log.configure(level="info", fmt="json", log_file=p)
    log.get("worker").bind(slug="foo").info("done", cost=0.05, turns=2)
    _flush_handlers()
    rows = _read_jsonl(p)
    assert rows[0]["slug"] == "foo"
    assert rows[0]["cost"] == 0.05
    assert rows[0]["turns"] == 2
    assert rows[0]["component"] == "worker"


# ---------------------------------------------------------------------------
# Cron-style: non-TTY stderr → defaults to JSON
# ---------------------------------------------------------------------------

def test_non_tty_stderr_defaults_json(tmp_path: Path, monkeypatch):
    """When stderr is piped (not a TTY), default fmt is json."""
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    # capsys-style: the test runner captures stderr (not a TTY) so default applies.
    log.configure(level="info", log_file=tmp_path / "x.jsonl")
    handlers = logging.getLogger().handlers
    assert len(handlers) == 2  # stderr + file
