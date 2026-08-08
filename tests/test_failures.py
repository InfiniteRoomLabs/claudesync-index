"""Failures log: append-only, atomic, count() for cron decision."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from reindex import failures


@pytest.fixture
def fail_log(tmp_path: Path, monkeypatch):
    p = tmp_path / "failures.jsonl"
    monkeypatch.setenv("CSINDEX_FAILURE_LOG", str(p))
    failures.truncate()
    return p


def test_record_appends_jsonl(fail_log: Path):
    failures.record(step="leaf", slug="x", kind="backend_failed", detail="boom")
    rec = json.loads(fail_log.read_text(encoding="utf-8").strip())
    assert rec["step"] == "leaf"
    assert rec["kind"] == "backend_failed"
    assert "ts" in rec


def test_count_returns_zero_when_empty(fail_log: Path):
    assert failures.count() == 0


def test_count_after_multiple(fail_log: Path):
    for i in range(5):
        failures.record(step="leaf", slug=f"s{i}", kind="schema_violation")
    assert failures.count() == 5


def test_count_skips_blank_lines(fail_log: Path):
    fail_log.write_text("\n\n\n", encoding="utf-8")
    assert failures.count() == 0


def test_record_truncates_long_detail(fail_log: Path):
    failures.record(step="leaf", slug="x", kind="oops", detail="A" * 5000)
    rec = json.loads(fail_log.read_text(encoding="utf-8").strip())
    assert len(rec["detail"]) <= 500


def test_record_stores_structured_context(fail_log: Path):
    failures.record(
        step="leaf",
        slug="x",
        kind="backend_failed",
        detail="claude -p exit 1",
        context={
            "kind": "process_exit",
            "exit_code": 1,
            "stderr": "auth failure: token expired",
            "stdout": "",
        },
    )
    rec = json.loads(fail_log.read_text(encoding="utf-8").strip())
    assert rec["context"]["kind"] == "process_exit"
    assert rec["context"]["exit_code"] == 1
    assert rec["context"]["stderr"] == "auth failure: token expired"


def test_record_caps_long_stderr_in_context(fail_log: Path):
    failures.record(
        step="leaf", slug="x", kind="oops",
        context={"stderr": "X" * 10_000, "exit_code": 1},
    )
    rec = json.loads(fail_log.read_text(encoding="utf-8").strip())
    assert len(rec["context"]["stderr"]) <= 4096
    # Non-string values pass through unchanged.
    assert rec["context"]["exit_code"] == 1


def test_record_no_log_path_silent(monkeypatch):
    monkeypatch.delenv("CSINDEX_FAILURE_LOG", raising=False)
    failures.record(step="leaf", slug="x", kind="oops")  # no exception


def test_count_no_log_path(monkeypatch):
    monkeypatch.delenv("CSINDEX_FAILURE_LOG", raising=False)
    assert failures.count() == 0


def test_concurrent_record_atomic(fail_log: Path):
    def worker(i: int):
        failures.record(step="leaf", slug=f"s{i}", kind="x", detail="d")
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert failures.count() == 20
