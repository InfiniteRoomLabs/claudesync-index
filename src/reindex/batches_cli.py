"""
`csindex batches` subcommands for inspecting and managing pending batch state.

Output streams:
  stdout — DATA (tables for humans, ndjson for --json). What the user asked for.
  stderr — LOGS (structured events). What the tool is doing.

Commands:
  list     show pending batches (--live to fetch server status, --json for ndjson)
  show     inspect a single batch
  cancel   cancel server-side + remove from local state
  purge    remove batches that 404 server-side
  resume   drain pending state (poll, retrieve, finalize)

Read-only (list, show) skip the lockfile. State-mutating commands acquire it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from reindex import config, cost_log, exit_codes, failures, log, models, paths, shutdown, workers
from reindex.lockfile import single_instance
from reindex.providers import BatchCapable, get_provider, provider_class
from reindex.state import BatchState, PersistedBatch

app = typer.Typer(no_args_is_help=True, add_completion=True)


def _batch_provider(name: str | config.ProviderName = config.ProviderName.ANTHROPIC) -> BatchCapable:
    """Instantiate the batch-capable provider that owns a persisted batch.

    Batch management talks to the provider named in local state
    (PersistedBatch.provider, default anthropic) — NOT the run-time
    default provider, which may have no batch mode at all.
    """
    cls = provider_class(name)
    if not issubclass(cls, BatchCapable):
        log.get("batches").error("provider_not_batch_capable", provider=str(name))
        raise typer.Exit(exit_codes.USAGE)
    return get_provider(name, config.load(paths.EXPORT_ROOT, provider_name=name))  # type: ignore[return-value]


def _complete_batch_id(incomplete: str):
    """Yield pending batch IDs from local state for shell completion."""
    try:
        pending = BatchState(paths.EXPORT_ROOT).load()
    except Exception:
        return
    for pb in pending:
        if pb.batch_id.startswith(incomplete):
            yield pb.batch_id

# stdout console for user-facing data; stderr is owned by the structlog stream.
out = Console(file=sys.stdout, soft_wrap=True)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _setup(
    log_level: str | None,
    log_format: str | None,
    no_color: bool,
    *,
    need_api_key: bool = False,
    log_file: str | None = None,
    no_log_file: bool = False,
) -> None:
    """Configure logging + dotenv. Logs go to stderr; this fn does not touch stdout."""
    if no_color:
        os.environ["NO_COLOR"] = "1"
    load_dotenv(paths.EXPORT_ROOT / ".env")
    os.environ["CSINDEX_ROOT"] = str(paths.EXPORT_ROOT)
    log.configure(
        level=log_level or "warn",  # quiet by default for management commands
        fmt=log_format,
        log_file=log_file,
        no_log_file=no_log_file,
    )
    if need_api_key and not os.environ.get("ANTHROPIC_API_KEY"):
        log.get("batches").error("missing_api_key", hint="set ANTHROPIC_API_KEY in .env")
        raise typer.Exit(exit_codes.CONFIG)


def _is_json_mode(json_flag: bool) -> bool:
    """JSON output if --json passed OR stdout isn't a TTY (cron/pipe friendly)."""
    return json_flag or not sys.stdout.isatty()


def _emit_json(records: list[dict]) -> None:
    """Emit one JSON object per line on stdout."""
    for r in records:
        sys.stdout.write(json.dumps(r) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@app.command(name="list")
def list_batches(
    root: Annotated[Path | None, typer.Option(
        "--root", help="Claudesync export tree (default: $CSINDEX_ROOT, then CWD). "
        "Also accepted before the subcommand: `csindex --root PATH batches list ...`.",
    )] = None,
    live: Annotated[bool, typer.Option("--live", help="Fetch live status from Anthropic.")] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Emit ndjson to stdout instead of a table.")] = False,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    log_format: Annotated[str | None, typer.Option("--log-format")] = None,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
) -> None:
    """List pending batches from local state."""
    if root is not None:
        paths.set_requested_root(root)
    from reindex.cli import _require_root_or_exit
    _require_root_or_exit()
    _setup(log_level, log_format, no_color, need_api_key=live)
    state = BatchState(paths.EXPORT_ROOT)
    pending = state.load()

    if not pending:
        if _is_json_mode(json_out):
            sys.stdout.write("")
        else:
            out.print("[dim]No pending batches.[/dim]")
        raise typer.Exit(exit_codes.OK)

    rows: list[dict] = [_row_for(pb) for pb in pending]

    if live:
        live_data = asyncio.run(_fetch_live(pending))
        for row, live_info in zip(rows, live_data, strict=True):
            row.update(live_info)

    if _is_json_mode(json_out):
        _emit_json(rows)
    else:
        _print_list_table(rows, with_live=live)

    raise typer.Exit(exit_codes.OK)


def _row_for(pb: PersistedBatch) -> dict:
    return {
        "batch_id": pb.batch_id,
        "step": pb.step,
        "model": pb.model,
        "items": len(pb.items),
        "submitted_at": pb.submitted_at,
        "is_retry": pb.is_retry,
    }


async def _fetch_live(pending: list[PersistedBatch]) -> list[dict]:
    providers = {pb.provider: _batch_provider(pb.provider) for pb in {pb.provider: pb for pb in pending}.values()}
    rows: list[dict] = []
    try:
        for pb in pending:
            try:
                st = await providers[pb.provider].batch_status(pb.batch_id)
                rows.append({"live_status": st.status, **st.counts})
            except Exception as e:
                rows.append({"live_status": "unavailable", "live_error": str(e)[:120]})
    finally:
        for p in providers.values():
            await p.aclose()
    return rows


def _print_list_table(rows: list[dict], *, with_live: bool) -> None:
    t = Table(
        title=f"{len(rows)} pending batch{'es' if len(rows) != 1 else ''}",
        title_style="bold",
        show_lines=False,
        pad_edge=False,
    )
    t.add_column("Batch ID", style="cyan", no_wrap=True)
    t.add_column("Step", style="magenta")
    t.add_column("Items", justify="right")
    t.add_column("Submitted", style="dim")
    t.add_column("Retry", justify="center")
    if with_live:
        t.add_column("Status")
        t.add_column("✓", justify="right", style="green")
        t.add_column("✗", justify="right", style="red")
        t.add_column("⏳", justify="right", style="yellow")
    for r in rows:
        cells = [
            r["batch_id"],
            r["step"],
            str(r["items"]),
            _short_ts(r["submitted_at"]),
            "↻" if r.get("is_retry") else "",
        ]
        if with_live:
            status = r.get("live_status", "?")
            status_styled = _style_status(status)
            cells += [
                status_styled,
                str(r.get("succeeded", "")),
                str(r.get("errored", "")),
                str(r.get("processing", "")),
            ]
        t.add_row(*cells)
    out.print(t)


def _short_ts(ts: str) -> str:
    """`2026-05-01T18:49:46Z` -> `05-01 18:49`."""
    try:
        if "T" in ts:
            date, rest = ts.split("T", 1)
            time = rest[:5]
            return f"{date[5:]} {time}"
    except Exception:
        pass
    return ts


def _style_status(s: str) -> str:
    if s == "ended":
        return "[green]ended[/green]"
    if s in ("in_progress",):
        return "[yellow]in_progress[/yellow]"
    if s == "canceling":
        return "[orange3]canceling[/orange3]"
    if s in ("expired", "unavailable"):
        return f"[red]{s}[/red]"
    return s


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

@app.command()
def show(
    batch_id: Annotated[str, typer.Argument(autocompletion=_complete_batch_id)],
    root: Annotated[Path | None, typer.Option(
        "--root", help="Claudesync export tree (default: $CSINDEX_ROOT, then CWD). "
        "Also accepted before the subcommand: `csindex --root PATH batches show ...`.",
    )] = None,
    live: Annotated[bool, typer.Option("--live/--no-live")] = True,
    json_out: Annotated[bool, typer.Option("--json")] = False,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    log_format: Annotated[str | None, typer.Option("--log-format")] = None,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
) -> None:
    """Show one batch — local items + (default) live status."""
    if root is not None:
        paths.set_requested_root(root)
    from reindex.cli import _require_root_or_exit
    _require_root_or_exit()
    _setup(log_level, log_format, no_color, need_api_key=live)
    state = BatchState(paths.EXPORT_ROOT)
    pending = state.load()
    pb = next((b for b in pending if b.batch_id == batch_id), None)
    if pb is None:
        log.get("batches").error("not_in_state", batch_id=batch_id)
        raise typer.Exit(exit_codes.USAGE)

    payload: dict = {
        "batch_id": pb.batch_id,
        "step": pb.step,
        "model": pb.model,
        "submitted_at": pb.submitted_at,
        "is_retry": pb.is_retry,
        "items": [{"custom_id": it.custom_id, **it.step_kwargs} for it in pb.items],
    }

    if live:
        try:
            payload["live"] = asyncio.run(_fetch_one_live(batch_id, pb.provider))
        except Exception as e:
            log.get("batches").error("live_fetch_failed", error=str(e)[:200])
            raise typer.Exit(exit_codes.UNAVAILABLE) from e

    if _is_json_mode(json_out):
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        _print_show(payload)

    raise typer.Exit(exit_codes.OK)


async def _fetch_one_live(batch_id: str, provider_name: str = "anthropic") -> dict:
    provider = _batch_provider(provider_name)
    try:
        st = await provider.batch_status(batch_id)
        return {
            "processing_status": st.status,
            **st.counts,
            "expires_at": st.expires_at,
        }
    finally:
        await provider.aclose()


def _print_show(payload: dict) -> None:
    out.print(f"[bold cyan]{payload['batch_id']}[/bold cyan]")
    out.print(f"  step:       {payload['step']}")
    out.print(f"  model:      {payload['model']}")
    out.print(f"  submitted:  {payload['submitted_at']}")
    out.print(f"  retry:      {'yes' if payload['is_retry'] else 'no'}")
    out.print(f"  items:      {len(payload['items'])}")

    if "live" in payload:
        live = payload["live"]
        out.print()
        out.print("[bold]Live status[/bold]")
        out.print(f"  status:     {_style_status(live['processing_status'])}")
        out.print(f"  succeeded:  [green]{live['succeeded']}[/green]")
        out.print(f"  errored:    [red]{live['errored']}[/red]")
        out.print(f"  processing: [yellow]{live['processing']}[/yellow]")
        out.print(f"  canceled:   {live['canceled']}")
        out.print(f"  expired:    {live['expired']}")
        out.print(f"  expires_at: [dim]{live['expires_at']}[/dim]")

    if payload["items"]:
        out.print()
        t = Table(title=f"{len(payload['items'])} items", show_lines=False, pad_edge=False)
        # Collect all kwarg keys present.
        kw_keys: list[str] = []
        for it in payload["items"]:
            for k in it.keys():
                if k != "custom_id" and k not in kw_keys:
                    kw_keys.append(k)
        t.add_column("custom_id", style="cyan", no_wrap=True)
        for k in kw_keys:
            t.add_column(k, style="dim")
        for it in payload["items"]:
            t.add_row(it["custom_id"], *(_truncate(str(it.get(k, ""))) for k in kw_keys))
        out.print(t)


def _truncate(s: str, n: int = 40) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------

@app.command()
def cancel(
    batch_id: Annotated[str, typer.Argument(autocompletion=_complete_batch_id)],
    root: Annotated[Path | None, typer.Option(
        "--root", help="Claudesync export tree (default: $CSINDEX_ROOT, then CWD). "
        "Also accepted before the subcommand: `csindex --root PATH batches cancel ...`.",
    )] = None,
    keep_state: Annotated[bool, typer.Option(
        "--keep-state", help="Don't remove from local state after cancel call.",
    )] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    log_format: Annotated[str | None, typer.Option("--log-format")] = None,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
) -> None:
    """Cancel a batch on Anthropic and remove it from local state."""
    if root is not None:
        paths.set_requested_root(root)
    from reindex.cli import _require_root_or_exit
    _require_root_or_exit()
    _setup(log_level, log_format, no_color, need_api_key=True)
    state = BatchState(paths.EXPORT_ROOT)
    pending = state.load()
    pb = next((b for b in pending if b.batch_id == batch_id), None)

    result: dict = {"batch_id": batch_id, "in_local_state": pb is not None}

    async def _do_cancel():
        provider = _batch_provider(pb.provider if pb is not None else config.ProviderName.ANTHROPIC)
        try:
            return await provider.batch_cancel(batch_id)
        finally:
            await provider.aclose()

    try:
        with single_instance(paths.EXPORT_ROOT):
            try:
                result["server_status"] = asyncio.run(_do_cancel())
                result["server_cancel"] = "ok"
            except Exception as e:
                result["server_cancel"] = "failed"
                result["server_error"] = str(e)[:200]

            if not keep_state and pb is not None:
                state.remove(batch_id)
                result["local_removed"] = True
            else:
                result["local_removed"] = False
    except RuntimeError as e:
        log.get("batches").error("lock_failed", error=str(e))
        raise typer.Exit(exit_codes.TEMPFAIL) from e

    if _is_json_mode(json_out):
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    else:
        _print_cancel(result)
    raise typer.Exit(exit_codes.OK)


def _print_cancel(r: dict) -> None:
    out.print(f"[cyan]{r['batch_id']}[/cyan]")
    if r["server_cancel"] == "ok":
        out.print(f"  server: [green]✓ cancel requested[/green] (status={r.get('server_status', '?')})")
    else:
        out.print(f"  server: [red]✗ cancel failed[/red] — {r.get('server_error', 'unknown')}")
    if r["in_local_state"]:
        if r["local_removed"]:
            out.print("  local:  [green]✓ removed from state[/green]")
        else:
            out.print("  local:  [dim]kept in state (--keep-state)[/dim]")
    else:
        out.print("  local:  [dim]not in local state[/dim]")


# ---------------------------------------------------------------------------
# purge
# ---------------------------------------------------------------------------

@app.command()
def purge(
    root: Annotated[Path | None, typer.Option(
        "--root", help="Claudesync export tree (default: $CSINDEX_ROOT, then CWD). "
        "Also accepted before the subcommand: `csindex --root PATH batches purge ...`.",
    )] = None,
    json_out: Annotated[bool, typer.Option("--json")] = False,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    log_format: Annotated[str | None, typer.Option("--log-format")] = None,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
) -> None:
    """Remove batches from local state that no longer exist on Anthropic (404)."""
    if root is not None:
        paths.set_requested_root(root)
    from reindex.cli import _require_root_or_exit
    _require_root_or_exit()
    _setup(log_level, log_format, no_color, need_api_key=True)
    state = BatchState(paths.EXPORT_ROOT)
    pending = state.load()
    if not pending:
        result = {"removed": [], "kept": [], "total": 0}
        if _is_json_mode(json_out):
            sys.stdout.write(json.dumps(result) + "\n")
        else:
            out.print("[dim]Nothing to purge.[/dim]")
        raise typer.Exit(exit_codes.OK)

    async def _check():
        providers = {pb.provider: _batch_provider(pb.provider) for pb in {pb.provider: pb for pb in pending}.values()}
        dead: list[tuple[str, str]] = []
        kept: list[str] = []
        try:
            for pb in pending:
                try:
                    if await providers[pb.provider].batch_exists(pb.batch_id):
                        kept.append(pb.batch_id)
                    else:
                        dead.append((pb.batch_id, "not found (404)"))
                except Exception as e:
                    # Transient errors keep the batch — purge only removes
                    # confirmed-dead entries.
                    log.get("batches").warn("purge_check_failed", batch_id=pb.batch_id, error=str(e)[:120])
                    kept.append(pb.batch_id)
            return dead, kept
        finally:
            for p in providers.values():
                await p.aclose()

    try:
        with single_instance(paths.EXPORT_ROOT):
            dead, kept = asyncio.run(_check())
            for bid, _ in dead:
                state.remove(bid)
    except RuntimeError as e:
        log.get("batches").error("lock_failed", error=str(e))
        raise typer.Exit(exit_codes.TEMPFAIL) from e

    result = {
        "removed": [{"batch_id": bid, "reason": reason} for bid, reason in dead],
        "kept": kept,
        "total": len(pending),
    }
    if _is_json_mode(json_out):
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    else:
        _print_purge(result)
    raise typer.Exit(exit_codes.OK)


def _print_purge(r: dict) -> None:
    out.print(
        f"Purged [red]{len(r['removed'])}[/red] / kept [green]{len(r['kept'])}[/green] "
        f"of {r['total']} pending"
    )
    for d in r["removed"]:
        out.print(f"  [red]✗[/red] {d['batch_id']} — [dim]{d['reason']}[/dim]")


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------

@app.command()
def resume(
    root: Annotated[Path | None, typer.Option(
        "--root", help="Claudesync export tree (default: $CSINDEX_ROOT, then CWD). "
        "Also accepted before the subcommand: `csindex --root PATH batches resume ...`.",
    )] = None,
    max_in_flight: Annotated[int, typer.Option(
        "--max-in-flight", help="Max concurrent batches polled at once. Default 4.",
    )] = 4,
    log_file: Annotated[str | None, typer.Option("--log-file")] = None,
    no_log_file: Annotated[bool, typer.Option("--no-log-file")] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    log_format: Annotated[str | None, typer.Option("--log-format")] = None,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
) -> None:
    """Drain pending state — poll, retrieve, finalize each pending batch."""
    if root is not None:
        paths.set_requested_root(root)
    from reindex.cli import _require_root_or_exit
    _require_root_or_exit()
    # Resume is a long-running pipeline op; default log level is info (verbose).
    _setup(log_level or "info", log_format, no_color, need_api_key=True,
           log_file=log_file, no_log_file=no_log_file)
    state = BatchState(paths.EXPORT_ROOT)
    if state.is_empty():
        if _is_json_mode(json_out):
            sys.stdout.write(json.dumps({"resumed": 0, "failures": 0}) + "\n")
        else:
            out.print("[dim]Nothing pending.[/dim]")
        raise typer.Exit(exit_codes.OK)

    pending_count = len(state.load())

    os.environ.setdefault("CSINDEX_COST_LOG", str(paths.EXPORT_ROOT / ".reindex-costs.jsonl"))
    os.environ.setdefault("CSINDEX_FAILURE_LOG", str(paths.EXPORT_ROOT / ".reindex-failures.jsonl"))
    cost_log.truncate()
    failures.truncate()

    async def _drain():
        shutdown.install(asyncio.get_running_loop())
        # One resume_pending call per provider that owns pending batches.
        names = sorted({pb.provider for pb in state.load()})
        for name in names:
            provider = _batch_provider(name)
            try:
                await provider.resume_pending(
                    state=state,
                    schema_for_step=lambda s: models.STEP_MODEL[s],
                    finalize_persisted=workers.finalize_persisted,
                    max_in_flight=max_in_flight,
                )
            finally:
                await provider.aclose()

    exit_code = exit_codes.OK
    try:
        with single_instance(paths.EXPORT_ROOT):
            try:
                asyncio.run(_drain())
            except Exception as e:
                log.get("batches").error("resume_failed", error=str(e)[:300])
                exit_code = exit_codes.SOFTWARE
    except RuntimeError as e:
        log.get("batches").error("lock_failed", error=str(e))
        raise typer.Exit(exit_codes.TEMPFAIL) from e

    fail_n = failures.count()
    cost_agg = cost_log.aggregate()
    if fail_n > 0 and exit_code == exit_codes.OK:
        exit_code = exit_codes.TEMPFAIL

    summary = {
        "resumed_batches": pending_count,
        "completed_calls": cost_agg["n"],
        "cost_usd": cost_agg["cost"],
        "failures": fail_n,
        "exit_code": exit_code,
    }

    if _is_json_mode(json_out):
        sys.stdout.write(json.dumps(summary) + "\n")
    else:
        _print_resume(summary)
    raise typer.Exit(exit_code)


def _print_resume(s: dict) -> None:
    out.print(f"Resumed [bold]{s['resumed_batches']}[/bold] batch(es)")
    out.print(f"  completed: [green]{s['completed_calls']}[/green] calls")
    out.print(f"  cost:      [yellow]${s['cost_usd']:.4f}[/yellow]")
    out.print(f"  failures:  {'[red]' + str(s['failures']) + '[/red]' if s['failures'] else '0'}")
    out.print(f"  exit:      {s['exit_code']}")
