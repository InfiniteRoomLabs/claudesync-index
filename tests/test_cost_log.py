"""Cost log: append, aggregate, pricing."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from reindex import cost_log


@pytest.fixture
def cost_log_path(tmp_path: Path, monkeypatch):
    p = tmp_path / "costs.jsonl"
    monkeypatch.setenv("CSINDEX_COST_LOG", str(p))
    cost_log.truncate()
    return p


def test_record_writes_jsonl(cost_log_path: Path):
    cost_log.record(step="leaf", slug="foo", cost=0.05, turns=2, duration_ms=1000, model="claude-haiku-4-5-20251001")
    lines = cost_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["slug"] == "foo"
    assert rec["cost"] == 0.05
    assert rec["model"] == "claude-haiku-4-5-20251001"
    assert "ts" in rec


def test_record_no_log_path_silent(monkeypatch):
    monkeypatch.delenv("CSINDEX_COST_LOG", raising=False)
    cost_log.record(step="leaf", slug="x", cost=0.1, turns=1, duration_ms=1, model="m")  # no exception


def test_aggregate_filters_by_step(cost_log_path: Path):
    cost_log.record(step="leaf", slug="a", cost=0.1, turns=1, duration_ms=100, model="m")
    cost_log.record(step="leaf", slug="b", cost=0.2, turns=2, duration_ms=200, model="m")
    cost_log.record(step="project", slug="p1", cost=1.0, turns=1, duration_ms=500, model="m")

    leaf = cost_log.aggregate(step="leaf")
    assert leaf["n"] == 2
    assert leaf["cost"] == pytest.approx(0.3)
    assert leaf["turns"] == 3
    assert leaf["wall_ms"] == 300

    proj = cost_log.aggregate(step="project")
    assert proj["n"] == 1


def test_aggregate_grand_total(cost_log_path: Path):
    cost_log.record(step="leaf", slug="a", cost=0.1, turns=1, duration_ms=100, model="m")
    cost_log.record(step="project", slug="p", cost=1.0, turns=1, duration_ms=500, model="m")
    cost_log.record(step="root", slug="root", cost=5.0, turns=1, duration_ms=100, model="m")

    grand = cost_log.aggregate()
    assert grand["n"] == 3
    assert grand["cost"] == pytest.approx(6.1)


def test_aggregate_empty(cost_log_path: Path):
    agg = cost_log.aggregate()
    assert agg == {"n": 0, "cost": 0.0, "turns": 0, "wall_ms": 0}


def test_aggregate_skips_blank_lines(cost_log_path: Path):
    cost_log_path.write_text("\n\n\n", encoding="utf-8")
    assert cost_log.aggregate()["n"] == 0


def test_aggregate_tolerates_corrupt_lines(cost_log_path: Path):
    cost_log_path.write_text(
        '{"step":"leaf","slug":"a","cost":0.1,"turns":1,"duration_ms":1,"model":"m"}\n'
        "not json at all\n"
        '{"step":"leaf","slug":"b","cost":0.2,"turns":1,"duration_ms":1,"model":"m"}\n',
        encoding="utf-8",
    )
    agg = cost_log.aggregate()
    # Corrupt line skipped; valid records counted.
    assert agg["n"] == 2
    assert agg["cost"] == pytest.approx(0.3)


def test_truncate_clears_file(cost_log_path: Path):
    cost_log.record(step="leaf", slug="x", cost=0.1, turns=1, duration_ms=1, model="m")
    cost_log.truncate()
    assert cost_log_path.read_text(encoding="utf-8") == ""


def test_compute_cost_haiku():
    # Haiku: $1/M in, $5/M out
    cost = cost_log.compute_cost("claude-haiku-4-5-20251001", 1_000_000, 0)
    assert cost == pytest.approx(1.0)
    cost = cost_log.compute_cost("claude-haiku-4-5-20251001", 0, 1_000_000)
    assert cost == pytest.approx(5.0)


def test_compute_cost_sonnet():
    cost = cost_log.compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000)
    assert cost == pytest.approx(3.0 + 15.0)


def test_compute_cost_opus():
    cost = cost_log.compute_cost("claude-opus-4-7", 100, 100)
    expected = (100 * 15 + 100 * 75) / 1_000_000
    assert cost == pytest.approx(expected)


def test_compute_cost_unknown_model_returns_zero():
    assert cost_log.compute_cost("not-a-real-model", 1000, 1000) == 0.0


def test_compute_cost_handles_versioned_haiku():
    """Pricing key is a prefix; specific date suffixes still match."""
    a = cost_log.compute_cost("claude-haiku-4-5", 1000, 1000)
    b = cost_log.compute_cost("claude-haiku-4-5-20251001", 1000, 1000)
    assert a == b


def test_concurrent_writes_atomic(cost_log_path: Path):
    """20 threads writing simultaneously should not lose any record (lines <4KB → atomic append)."""
    def worker(i: int):
        cost_log.record(step="leaf", slug=f"s{i}", cost=0.01, turns=1, duration_ms=1, model="m")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert cost_log.aggregate()["n"] == 20
