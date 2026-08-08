"""Tests for `reindex batches` subcommands."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from reindex import batches_cli, cli, exit_codes
from reindex.lockfile import single_instance
from reindex.state import BatchState, PersistedItem

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_state(tmp_export, batch_id="msgbatch_a", step="leaf", items_count=2):
    s = BatchState(tmp_export)
    s.add(
        batch_id=batch_id,
        step=step,
        model="claude-haiku-4-5",
        items=[
            PersistedItem(custom_id=f"id{i}", step_kwargs={"slug": f"id{i}"})
            for i in range(items_count)
        ],
    )
    return s


def _patch_dotenv(monkeypatch):
    monkeypatch.setattr(batches_cli, "load_dotenv", lambda *a, **k: None)


def _mock_provider():
    """Provider-method mock returned by batches_cli._batch_provider."""
    from reindex.providers.base import BatchStatus
    prov = MagicMock()
    prov.batch_status = AsyncMock(return_value=BatchStatus(
        status="ended", done=True,
        counts={"succeeded": 2, "errored": 0, "processing": 0, "expired": 0, "canceled": 0},
        expires_at="2026-12-01T00:00:00Z",
    ))
    prov.batch_cancel = AsyncMock(return_value="canceling")
    prov.batch_exists = AsyncMock(return_value=True)
    prov.resume_pending = AsyncMock()
    prov.aclose = AsyncMock()
    return prov


def _patch_provider(prov):
    return patch("reindex.batches_cli._batch_provider", return_value=prov)



# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def test_list_no_pending(tmp_export, monkeypatch):
    _patch_dotenv(monkeypatch)
    res = runner.invoke(cli.app, ["batches", "list"])
    assert res.exit_code == exit_codes.OK


def test_list_pending_local_only(tmp_export, monkeypatch):
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export)
    res = runner.invoke(cli.app, ["batches", "list", "--json"])
    assert res.exit_code == exit_codes.OK


def test_list_root_after_subcommand(tmp_export, monkeypatch):
    """`--root` placed after the subcommand must parse (command-level option), not exit 2."""
    _patch_dotenv(monkeypatch)
    res = runner.invoke(cli.app, ["batches", "list", "--root", str(tmp_export)])
    assert res.exit_code == exit_codes.OK


def test_list_live_fetches_status(tmp_export, fake_api_key, monkeypatch):
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export)

    prov = _mock_provider()
    with _patch_provider(prov):
        res = runner.invoke(cli.app, ["batches", "list", "--live"])
    assert res.exit_code == exit_codes.OK
    prov.batch_status.assert_awaited_once()


def test_list_live_handles_retrieve_failure(tmp_export, fake_api_key, monkeypatch):
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export)

    prov = _mock_provider()
    prov.batch_status = AsyncMock(side_effect=Exception("404 not found"))
    with _patch_provider(prov):
        res = runner.invoke(cli.app, ["batches", "list", "--live"])
    assert res.exit_code == exit_codes.OK  # warns but doesn't fail


def test_list_live_without_api_key_returns_config(tmp_export, monkeypatch):
    _patch_dotenv(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _seed_state(tmp_export)
    res = runner.invoke(cli.app, ["batches", "list", "--live"])
    assert res.exit_code == exit_codes.CONFIG


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

def test_show_unknown_batch_returns_usage(tmp_export, monkeypatch):
    _patch_dotenv(monkeypatch)
    res = runner.invoke(cli.app, ["batches", "show", "msgbatch_does_not_exist", "--no-live"])
    assert res.exit_code == exit_codes.USAGE


def test_show_local_only(tmp_export, monkeypatch):
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export)
    res = runner.invoke(cli.app, ["batches", "show", "msgbatch_a", "--no-live"])
    assert res.exit_code == exit_codes.OK


def test_show_with_live(tmp_export, fake_api_key, monkeypatch):
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export)

    with _patch_provider(_mock_provider()):
        res = runner.invoke(cli.app, ["batches", "show", "msgbatch_a"])
    assert res.exit_code == exit_codes.OK


def test_show_live_fetch_failure_returns_unavailable(tmp_export, fake_api_key, monkeypatch):
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export)

    prov = _mock_provider()
    prov.batch_status = AsyncMock(side_effect=Exception("oops"))
    with _patch_provider(prov):
        res = runner.invoke(cli.app, ["batches", "show", "msgbatch_a"])
    assert res.exit_code == exit_codes.UNAVAILABLE


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------

def test_cancel_removes_from_state(tmp_export, fake_api_key, monkeypatch):
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export)

    with _patch_provider(_mock_provider()):
        res = runner.invoke(cli.app, ["batches", "cancel", "msgbatch_a"])
    assert res.exit_code == exit_codes.OK
    assert BatchState(tmp_export).is_empty()


def test_cancel_keeps_state_when_flag_set(tmp_export, fake_api_key, monkeypatch):
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export)

    with _patch_provider(_mock_provider()):
        res = runner.invoke(cli.app, ["batches", "cancel", "msgbatch_a", "--keep-state"])
    assert res.exit_code == exit_codes.OK
    assert not BatchState(tmp_export).is_empty()


def test_cancel_unknown_batch_still_calls_server(tmp_export, fake_api_key, monkeypatch):
    """User can cancel a batch that isn't in local state (e.g. submitted from another machine)."""
    _patch_dotenv(monkeypatch)

    prov = _mock_provider()
    with _patch_provider(prov):
        res = runner.invoke(cli.app, ["batches", "cancel", "msgbatch_external"])
    assert res.exit_code == exit_codes.OK
    prov.batch_cancel.assert_awaited_once_with("msgbatch_external")


def test_cancel_server_failure_still_removes_local(tmp_export, fake_api_key, monkeypatch):
    """If server-side cancel fails (already done?), still remove from local state."""
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export)

    prov = _mock_provider()
    prov.batch_cancel = AsyncMock(side_effect=Exception("already_finalized"))
    with _patch_provider(prov):
        res = runner.invoke(cli.app, ["batches", "cancel", "msgbatch_a"])
    assert res.exit_code == exit_codes.OK
    assert BatchState(tmp_export).is_empty()


def test_cancel_lock_contention(tmp_export, fake_api_key, monkeypatch):
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export)
    with single_instance(tmp_export):
        res = runner.invoke(cli.app, ["batches", "cancel", "msgbatch_a"])
    assert res.exit_code == exit_codes.TEMPFAIL


# ---------------------------------------------------------------------------
# purge
# ---------------------------------------------------------------------------

def test_purge_nothing_pending(tmp_export, fake_api_key, monkeypatch):
    _patch_dotenv(monkeypatch)
    res = runner.invoke(cli.app, ["batches", "purge"])
    assert res.exit_code == exit_codes.OK


def test_purge_removes_404_batches(tmp_export, fake_api_key, monkeypatch):
    _patch_dotenv(monkeypatch)
    s = BatchState(tmp_export)
    s.add(batch_id="msgbatch_alive", step="leaf", model="m", items=[PersistedItem("a", {})])
    s.add(batch_id="msgbatch_dead", step="leaf", model="m", items=[PersistedItem("b", {})])

    prov = _mock_provider()
    prov.batch_exists = AsyncMock(side_effect=lambda bid: bid != "msgbatch_dead")
    with _patch_provider(prov):
        res = runner.invoke(cli.app, ["batches", "purge"])
    assert res.exit_code == exit_codes.OK

    remaining = {b.batch_id for b in BatchState(tmp_export).load()}
    assert remaining == {"msgbatch_alive"}


def test_purge_keeps_all_when_all_alive(tmp_export, fake_api_key, monkeypatch):
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export, batch_id="msgbatch_alive")

    with _patch_provider(_mock_provider()):
        res = runner.invoke(cli.app, ["batches", "purge"])
    assert res.exit_code == exit_codes.OK
    assert len(BatchState(tmp_export).load()) == 1


def test_purge_lock_contention(tmp_export, fake_api_key, monkeypatch):
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export)
    with single_instance(tmp_export):
        res = runner.invoke(cli.app, ["batches", "purge"])
    assert res.exit_code == exit_codes.TEMPFAIL


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------

def test_resume_nothing_pending(tmp_export, fake_api_key, monkeypatch):
    _patch_dotenv(monkeypatch)
    res = runner.invoke(cli.app, ["batches", "resume"])
    assert res.exit_code == exit_codes.OK


def test_resume_drains_state(tmp_export, fake_api_key, monkeypatch):
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export)

    prov = _mock_provider()
    with _patch_provider(prov):
        res = runner.invoke(cli.app, ["batches", "resume"])
    assert res.exit_code == exit_codes.OK
    prov.resume_pending.assert_awaited_once()


def test_resume_failures_return_tempfail(tmp_export, fake_api_key, monkeypatch):
    """If resume_pending records failures, exit with TEMPFAIL."""
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export)

    async def fake_resume(**kwargs):
        # Simulate one failure recorded mid-resume.
        from reindex import failures
        failures.record(step="leaf", slug="x", kind="test_failure")

    prov = _mock_provider()
    prov.resume_pending = fake_resume
    with _patch_provider(prov):
        res = runner.invoke(cli.app, ["batches", "resume"])
    assert res.exit_code == exit_codes.TEMPFAIL


def test_resume_unhandled_exception_returns_software(tmp_export, fake_api_key, monkeypatch):
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export)

    async def boom(**kwargs):
        raise RuntimeError("unhandled")

    prov = _mock_provider()
    prov.resume_pending = boom
    with _patch_provider(prov):
        res = runner.invoke(cli.app, ["batches", "resume"])
    assert res.exit_code == exit_codes.SOFTWARE


def test_resume_installs_shutdown_handler(tmp_export, fake_api_key, monkeypatch):
    """shutdown.install() must be called inside _drain() so SIGTERM is handled."""
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export)

    prov = _mock_provider()
    installed_loops: list = []

    def fake_install(loop):
        installed_loops.append(loop)
        from reindex.shutdown import ShutdownController
        return ShutdownController()

    with _patch_provider(prov), patch("reindex.batches_cli.shutdown.install", side_effect=fake_install):
        runner.invoke(cli.app, ["batches", "resume"])

    assert len(installed_loops) == 1, "shutdown.install() was not called during batches resume"


def test_resume_lock_contention(tmp_export, fake_api_key, monkeypatch):
    _patch_dotenv(monkeypatch)
    _seed_state(tmp_export)
    with single_instance(tmp_export):
        res = runner.invoke(cli.app, ["batches", "resume"])
    assert res.exit_code == exit_codes.TEMPFAIL


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

def test_batches_help_lists_subcommands():
    res = runner.invoke(cli.app, ["batches", "--help"])
    assert res.exit_code == 0
    for cmd in ["list", "show", "cancel", "purge", "resume"]:
        assert cmd in res.stdout
