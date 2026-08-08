"""CLI: argument dispatch, exit codes, lockfile integration, end-to-end smoke."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from reindex import cli, exit_codes
from reindex.lockfile import single_instance

runner = CliRunner()


# ---------------------------------------------------------------------------
# Help renders
# ---------------------------------------------------------------------------

def test_full_help():
    res = runner.invoke(cli.app, ["full", "--help"])
    assert res.exit_code == 0
    assert "--batch-size" in res.stdout
    assert "--api" in res.stdout


# ---------------------------------------------------------------------------
# Exit codes for argument errors
# ---------------------------------------------------------------------------

def test_typer_invalid_int_exit():
    res = runner.invoke(cli.app, ["full", "--batch-size", "foo"])
    # typer/click default for parse error is 2 (close enough to USAGE).
    assert res.exit_code != 0


def test_multiple_only_flags_exit_usage(tmp_export):
    res = runner.invoke(cli.app, ["full", "--only-leaves", "--only-projects"])
    assert res.exit_code == exit_codes.USAGE


def test_no_projects_with_only_projects_exit_usage(tmp_export):
    res = runner.invoke(cli.app, ["full", "--no-projects", "--only-projects"])
    assert res.exit_code == exit_codes.USAGE


# ---------------------------------------------------------------------------
# Auth + preflight
# ---------------------------------------------------------------------------

def test_api_without_key_exits_config(tmp_export, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Block dotenv from loading anything.
    monkeypatch.setenv("DOTENV_PATH_OVERRIDE", "/dev/null")
    # Patch load_dotenv to no-op so .env in repo doesn't restore the key.
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    res = runner.invoke(cli.app, ["full", "--api"])
    assert res.exit_code == exit_codes.CONFIG


# ---------------------------------------------------------------------------
# Lockfile: second run fails fast with TEMPFAIL
# ---------------------------------------------------------------------------

def test_concurrent_run_returns_tempfail(tmp_export, monkeypatch):
    """Hold the lock, then invoke full → should exit TEMPFAIL."""
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    with single_instance(tmp_export):
        res = runner.invoke(cli.app, ["full", "--no-projects", "--only-root"])
    assert res.exit_code == exit_codes.TEMPFAIL


# ---------------------------------------------------------------------------
# End-to-end happy path: subscription, single leaf
# ---------------------------------------------------------------------------

def test_full_subscription_e2e_single_leaf(tmp_export, make_conv, valid_leaf, monkeypatch):
    """Mock subscription_invoke, run pipeline, verify INDEX.md written + cost logged + exit 0."""
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    conv = make_conv("alpha", "## Human\n_2024-01-01_\n\nhi\n")

    # Build a successful InvokeResult that the pipeline will consume.
    from reindex.providers.base import InvokeResult
    fake_result = InvokeResult(
        payload=valid_leaf, cost=0.01, turns=1, duration_ms=100,
    )

    with patch("reindex.providers.claude_cli.ClaudeCliProvider.invoke",
               AsyncMock(return_value=fake_result)):
        res = runner.invoke(cli.app, [
            "full", "--no-projects", "--only-leaves",
            "--batch-size", "1", "--json",
        ])

    assert res.exit_code == exit_codes.OK, res.stdout
    assert (conv / "INDEX.md").is_file()
    content = (conv / "INDEX.md").read_text(encoding="utf-8")
    assert "STAMPED_AFTER" not in content
    assert "slug: alpha" in content


def test_full_partial_failure_returns_tempfail(tmp_export, make_conv, monkeypatch):
    """If any worker errors, exit code is TEMPFAIL so cron retries."""
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    make_conv("a", "## Human\n_2024-01-01_\n\nhi\n")

    with patch("reindex.providers.claude_cli.ClaudeCliProvider.invoke",
               AsyncMock(side_effect=RuntimeError("upstream down"))):
        res = runner.invoke(cli.app, [
            "full", "--no-projects", "--only-leaves",
            "--batch-size", "1",
        ])

    assert res.exit_code == exit_codes.TEMPFAIL


def test_limit_auto_gates_to_leaves(tmp_export, make_conv, valid_leaf, monkeypatch):
    """--limit without --only-* should silently coerce to step-1-only."""
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    make_conv("a", "x")
    make_conv("b", "y")
    make_conv("c", "z")

    from reindex.providers.base import InvokeResult
    fake_result = InvokeResult(payload=valid_leaf, cost=0.01, turns=1, duration_ms=10)
    with patch("reindex.providers.claude_cli.ClaudeCliProvider.invoke",
               AsyncMock(return_value=fake_result)) as mock_invoke:
        res = runner.invoke(cli.app, [
            "full", "--no-projects", "--limit", "2",
            "--batch-size", "1",
        ])
    assert res.exit_code == exit_codes.OK
    # Only 2 leaves invoked despite 3 present.
    assert mock_invoke.await_count == 2


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_standalone_dirs_lists_alphabetically(tmp_export, make_conv):
    make_conv("zebra", "x")
    make_conv("alpha", "x")
    make_conv("mike", "x")
    dirs = cli._standalone_dirs(tmp_export)
    names = [d.name for d in dirs]
    assert names == ["alpha", "mike", "zebra"]


def test_nested_dirs_finds_project_conversations(tmp_export, make_conv, make_project):
    make_project("p1")
    make_conv("c1", "x", project="p1")
    make_conv("c2", "x", project="p1")
    dirs = cli._nested_dirs(tmp_export)
    assert {d.name for d in dirs} == {"c1", "c2"}


def test_project_dirs_alphabetical(tmp_export, make_project):
    make_project("zeta")
    make_project("alpha")
    dirs = cli._project_dirs(tmp_export)
    assert [d.name for d in dirs] == ["alpha", "zeta"]


def test_gather_leaves_respects_limit(tmp_export, make_conv):
    for x in "abc":
        make_conv(x, "x")
    dirs = cli._gather_leaves("standalone", limit=2)
    assert len(dirs) == 2


# ---------------------------------------------------------------------------
# API path dispatch (batch) — verify it routes through batch.run, not subscription
# ---------------------------------------------------------------------------

def test_full_api_dispatches_through_batch(tmp_export, make_conv, valid_leaf_dict, fake_api_key, monkeypatch):
    """--api should call batch.run instead of subscription_invoke."""
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    make_conv("alpha", "## Human\n_2024-01-01_\n\nhi\n")

    async def fake_batch_run(*, step, model, schema_cls, tasks, batch_size, client,
                              finalize, serialize_context, state, max_in_flight=4, wait=False):
        # Simulate batch processing each task with the valid payload.
        for t in tasks:
            payload = schema_cls.model_validate(valid_leaf_dict)
            await finalize(t, payload, 0.005, 1, 100)

    sub_mock = AsyncMock()
    with patch("reindex.providers.claude_cli.ClaudeCliProvider.invoke", sub_mock), \
         patch("reindex.batch.run", new=fake_batch_run):
        res = runner.invoke(cli.app, ["full", "--api", "--no-projects", "--only-leaves", "--batch-size", "5"])

    assert res.exit_code == exit_codes.OK
    sub_mock.assert_not_called()  # API path used; subscription bypassed


def test_full_api_resume_pending_runs_first(tmp_export, fake_api_key, monkeypatch):
    """--api triggers batch.resume_pending at startup if state file has pending batches."""
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)

    from reindex.state import BatchState, PersistedItem
    s = BatchState(tmp_export)
    s.add(
        batch_id="msgbatch_pending",
        step="leaf",
        model="claude-haiku-4-5",
        items=[PersistedItem(custom_id="dummy", step_kwargs={"slug": "dummy"})],
    )

    resume_mock = AsyncMock()

    async def noop_run(**kwargs):
        return None

    with patch("reindex.batch.resume_pending", resume_mock), \
         patch("reindex.batch.run", new=noop_run):
        res = runner.invoke(cli.app, ["full", "--api", "--no-projects", "--only-leaves"])

    assert res.exit_code == exit_codes.OK
    resume_mock.assert_called_once()


# ---------------------------------------------------------------------------
# Provider precedence + flag validation
# ---------------------------------------------------------------------------

def test_subscription_flag_overrides_provider_env(tmp_export, make_conv, valid_leaf, monkeypatch):
    """$CSINDEX_PROVIDER=anthropic in env, but --subscription forces claude-cli."""
    monkeypatch.setenv("CSINDEX_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    make_conv("alpha", "## Human\n_2024-01-01_\n\nhi\n")

    from reindex.providers.base import InvokeResult
    fake_result = InvokeResult(payload=valid_leaf, cost=0.01, turns=1, duration_ms=10)

    batch_mock = AsyncMock()
    with patch("reindex.providers.claude_cli.ClaudeCliProvider.invoke",
               AsyncMock(return_value=fake_result)) as sub_mock, \
         patch("reindex.batch.run", batch_mock):
        res = runner.invoke(cli.app, [
            "full", "--subscription", "--no-projects", "--only-leaves",
            "--batch-size", "1",
        ])
    assert res.exit_code == exit_codes.OK
    assert sub_mock.await_count == 1
    batch_mock.assert_not_called()


def test_provider_flag_unknown_name_errors(tmp_export, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    res = runner.invoke(cli.app, ["full", "--provider", "nonsense"])
    assert res.exit_code != exit_codes.OK


def test_api_and_subscription_together_rejected(tmp_export, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    res = runner.invoke(cli.app, ["full", "--api", "--subscription"])
    assert res.exit_code == exit_codes.USAGE


# ---------------------------------------------------------------------------
# --max-in-flight + --wait require --api
# ---------------------------------------------------------------------------

def test_max_in_flight_without_api_rejected(tmp_export, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    res = runner.invoke(cli.app, ["full", "--max-in-flight", "8"])
    assert res.exit_code == exit_codes.USAGE


def test_wait_without_api_rejected(tmp_export, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    res = runner.invoke(cli.app, ["full", "--wait"])
    assert res.exit_code == exit_codes.USAGE


def test_max_in_flight_with_api_accepted(tmp_export, fake_api_key, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)

    async def fake_run(**kwargs):
        # Verify the flag propagates.
        assert kwargs.get("max_in_flight") == 8
        assert kwargs.get("wait") is False  # default

    with patch("reindex.batch.run", new=fake_run):
        res = runner.invoke(cli.app, [
            "full", "--api", "--no-projects", "--only-leaves",
            "--max-in-flight", "8",
        ])
    # Either OK or no-op (no leaves to process); just shouldn't be USAGE.
    assert res.exit_code != exit_codes.USAGE


def test_wait_with_api_accepted(tmp_export, fake_api_key, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)

    received = {}

    async def fake_run(**kwargs):
        received["wait"] = kwargs.get("wait")

    with patch("reindex.batch.run", new=fake_run):
        res = runner.invoke(cli.app, [
            "full", "--api", "--no-projects", "--only-leaves", "--wait",
        ])
    assert res.exit_code != exit_codes.USAGE


# ---------------------------------------------------------------------------
# --log-file / --no-log-file
# ---------------------------------------------------------------------------

def test_log_file_default_path(tmp_export, monkeypatch, valid_leaf):
    """Default file logging writes to <export>/.reindex.log.jsonl."""
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("CSINDEX_LOG_FILE", raising=False)

    from reindex.providers.base import InvokeResult
    fake_result = InvokeResult(payload=valid_leaf, cost=0.01, turns=1, duration_ms=10)
    with patch("reindex.providers.claude_cli.ClaudeCliProvider.invoke",
               AsyncMock(return_value=fake_result)):
        res = runner.invoke(cli.app, [
            "full", "--no-projects", "--only-leaves", "--limit", "0",
            "--batch-size", "1",
        ])
    assert res.exit_code == exit_codes.OK
    log_path = tmp_export / ".reindex.log.jsonl"
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "logs_init" in content


def test_no_log_file_disables(tmp_export, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("CSINDEX_LOG_FILE", raising=False)
    runner.invoke(cli.app, [
        "full", "--no-projects", "--only-leaves", "--limit", "0",
        "--batch-size", "1", "--no-log-file",
    ])
    log_path = tmp_export / ".reindex.log.jsonl"
    assert not log_path.exists()


def test_explicit_log_file_path(tmp_export, monkeypatch, valid_leaf):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    custom = tmp_export / "custom.jsonl"

    from reindex.providers.base import InvokeResult
    fake_result = InvokeResult(payload=valid_leaf, cost=0.01, turns=1, duration_ms=10)
    with patch("reindex.providers.claude_cli.ClaudeCliProvider.invoke",
               AsyncMock(return_value=fake_result)):
        runner.invoke(cli.app, [
            "full", "--no-projects", "--only-leaves", "--limit", "0",
            "--batch-size", "1", "--log-file", str(custom),
        ])
    assert custom.exists()
    # Default path should NOT exist (overridden).
    assert not (tmp_export / ".reindex.log.jsonl").exists()


# ---------------------------------------------------------------------------
# Runtime root selection (--root / $CSINDEX_ROOT / CWD)
# ---------------------------------------------------------------------------

def test_full_root_flag_invalid_tree_exits_65(tmp_path):
    res = runner.invoke(cli.app, ["full", "--root", str(tmp_path), "--only-leaves", "--limit", "1"])
    assert res.exit_code == exit_codes.DATAERR


def test_help_works_outside_export_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(cli.app, ["full", "--help"]).exit_code == 0


# ---------------------------------------------------------------------------
# Embedding commands (embed / search / embed-migrate)
# ---------------------------------------------------------------------------

def test_embed_help_works_outside_export_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(cli.app, ["embed", "--help"]).exit_code == 0
    assert runner.invoke(cli.app, ["search", "--help"]).exit_code == 0
    assert runner.invoke(cli.app, ["embed-migrate", "--help"]).exit_code == 0


def test_embed_invalid_root_exits_65(tmp_path):
    result = runner.invoke(cli.app, ["embed", "--root", str(tmp_path)])
    assert result.exit_code == exit_codes.DATAERR


def test_search_invalid_root_exits_65(tmp_path):
    result = runner.invoke(cli.app, ["search", "anything", "--root", str(tmp_path)])
    assert result.exit_code == exit_codes.DATAERR


def test_embed_both_no_flags_exits_usage(tmp_export):
    """--no-conversations + --no-summaries is checked before CF creds, so no
    env setup needed here."""
    res = runner.invoke(cli.app, ["embed", "--no-conversations", "--no-summaries"])
    assert res.exit_code == exit_codes.USAGE


def test_embed_missing_cf_creds_exits_config(tmp_export, monkeypatch):
    monkeypatch.setenv("CSINDEX_EMBED_BACKEND", "cloudflare")
    monkeypatch.delenv("CF_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CF_API_TOKEN", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    res = runner.invoke(cli.app, ["embed"])
    assert res.exit_code == exit_codes.CONFIG
    assert "CF_ACCOUNT_ID" in res.output


def test_search_missing_cf_creds_exits_config(tmp_export, monkeypatch):
    monkeypatch.setenv("CSINDEX_EMBED_BACKEND", "cloudflare")
    monkeypatch.delenv("CF_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CF_API_TOKEN", raising=False)
    res = runner.invoke(cli.app, ["search", "anything"])
    assert res.exit_code == exit_codes.CONFIG
    assert "CF_ACCOUNT_ID" in res.output


def test_embed_e2e_success(tmp_export, make_conv, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("CSINDEX_EMBED_BACKEND", "cloudflare")
    monkeypatch.setenv("CF_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CF_API_TOKEN", "tok")
    make_conv("alpha")

    from reindex import embedding
    stats = embedding.EmbedStats(files=1, chunks=2, skipped=0, failed=0)
    corpus_mock = AsyncMock(return_value=stats)
    monkeypatch.setattr(embedding, "embed_corpus", corpus_mock)

    res = runner.invoke(cli.app, ["embed", "--no-summaries", "--persist", "custom-vdb"])
    assert res.exit_code == exit_codes.OK
    assert corpus_mock.await_count == 1
    # persist_dir resolved relative to the passed --persist value.
    assert corpus_mock.await_args is not None
    called_persist_dir = corpus_mock.await_args.args[1]
    assert called_persist_dir == Path("custom-vdb")


def test_embed_e2e_reports_failures_as_tempfail(tmp_export, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("CSINDEX_EMBED_BACKEND", "cloudflare")
    monkeypatch.setenv("CF_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CF_API_TOKEN", "tok")

    from reindex import embedding
    stats = embedding.EmbedStats(files=0, chunks=0, skipped=0, failed=1)
    monkeypatch.setattr(embedding, "embed_corpus", AsyncMock(return_value=stats))

    res = runner.invoke(cli.app, ["embed"])
    assert res.exit_code == exit_codes.TEMPFAIL


def test_search_e2e_success(tmp_export, monkeypatch):
    monkeypatch.setenv("CSINDEX_EMBED_BACKEND", "cloudflare")
    monkeypatch.setenv("CF_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CF_API_TOKEN", "tok")

    from reindex import embedding
    hits = [{"score": 0.876, "slug": "conversations/alpha", "kind": "conversation", "text": "hello world", "meta": {}}]
    search_mock = AsyncMock(return_value=hits)
    monkeypatch.setattr(embedding, "search", search_mock)

    res = runner.invoke(cli.app, ["search", "hello", "--k", "1"])
    assert res.exit_code == exit_codes.OK
    assert search_mock.await_count == 1
    assert "conversations/alpha" in res.stdout
    assert "hello world" in res.stdout


def test_embed_migrate_e2e_success(tmp_export, monkeypatch):
    from reindex import embedding
    monkeypatch.setattr(embedding, "backfill_kind", lambda persist_dir: 3)

    res = runner.invoke(cli.app, ["embed-migrate", "--persist", "custom-vdb"])
    assert res.exit_code == exit_codes.OK


def test_embed_unconfigured_backend_exits_78(tmp_export, monkeypatch):
    monkeypatch.delenv("CSINDEX_EMBED_BACKEND", raising=False)
    result = runner.invoke(cli.app, ["embed"])
    assert result.exit_code == exit_codes.CONFIG


def test_search_collection_identity_mismatch_exits_65(tmp_export, monkeypatch, tmp_path):
    """A collection embedded under one backend/model, queried under another,
    must fail loudly (exit 65) rather than silently return nonsense hits.
    Uses ollama (no creds needed) and never reaches the network: open_collection
    raises CollectionMismatch before search() embeds the query."""
    from reindex import embedding

    persist_dir = tmp_path / "vdb"
    embedding.open_collection(persist_dir, backend="ollama", model="m1")

    monkeypatch.setenv("CSINDEX_EMBED_BACKEND", "ollama")
    monkeypatch.setenv("CSINDEX_EMBED_MODEL", "m2")

    res = runner.invoke(cli.app, ["search", "anything", "--persist", str(persist_dir)])
    assert res.exit_code == exit_codes.DATAERR


def test_embed_collection_identity_mismatch_exits_65(tmp_export, monkeypatch, tmp_path):
    """Mirrors test_search_collection_identity_mismatch_exits_65 for the embed
    side: a store opened under one backend/model, embedded against another,
    must fail loudly (exit 65) rather than silently corrupting the vector
    space. Uses ollama (no creds needed) and never reaches the network:
    open_collection raises CollectionMismatch before embed_corpus embeds
    anything."""
    from reindex import embedding

    persist_dir = tmp_path / "vdb"
    embedding.open_collection(persist_dir, backend="ollama", model="m1")

    monkeypatch.setenv("CSINDEX_EMBED_BACKEND", "ollama")
    monkeypatch.setenv("CSINDEX_EMBED_MODEL", "m2")

    res = runner.invoke(cli.app, ["embed", "--persist", str(persist_dir)])
    assert res.exit_code == exit_codes.DATAERR


# ---------------------------------------------------------------------------
# quick: claude-cli refresh with packaged prompt
# ---------------------------------------------------------------------------

def test_quick_help_works_outside_export_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(cli.app, ["quick", "--help"]).exit_code == 0


def test_quick_invalid_root_exits_65(tmp_path):
    result = runner.invoke(cli.app, ["quick", "--root", str(tmp_path)])
    assert result.exit_code == exit_codes.DATAERR


def test_quick_missing_claude_binary_exits_78(tmp_export, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    result = runner.invoke(cli.app, ["quick"])
    assert result.exit_code == exit_codes.CONFIG


def test_quick_pipes_packaged_prompt_to_claude(tmp_export, monkeypatch):
    calls = {}

    def fake_run(argv, **kw):
        calls["argv"] = argv
        calls["input"] = kw.get("input", "")
        import subprocess
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr("subprocess.run", fake_run)
    result = runner.invoke(cli.app, ["quick"])
    assert result.exit_code == 0
    assert calls["argv"][0] == "claude"
    assert "{{EXPORT_DIR}}" not in calls["input"]
    assert "METADATA.json" in calls["input"]
