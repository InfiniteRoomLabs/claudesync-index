"""
csindex CLI — orchestrator for the depth-first index pipeline.

Provider selection (--provider NAME, default claude-cli):
  claude-cli   claude -p subprocess on the subscription quota; concurrent
               up to --batch-size. Alias: --subscription.
  anthropic    Anthropic Message Batches API; submits in chunks of
               --batch-size for 50% discount. Alias: --api.
  (others)     see `csindex full --help` / providers/_template.py.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from reindex import (
    batches_cli,
    config,
    cost_log,
    exit_codes,
    failures,
    log,
    models,
    paths,
    prompt_loader,
    repair,
    shutdown,
    workers,
)
from reindex import runner as runner_mod
from reindex.lockfile import single_instance
from reindex.providers import BatchCapable, Provider, get_provider, provider_class
from reindex.providers.base import ProviderFailure
from reindex.state import BatchState

app = typer.Typer(no_args_is_help=False, add_completion=True)
app.add_typer(batches_cli.app, name="batches", help="Inspect and manage pending Message Batches state.")
# repair.app is a single-command Typer app; add_typer would mount it as a
# group requiring a sub-subcommand (`repair-hashes main`). Register the
# callback directly instead so the surface is exactly `repair-hashes [--dry-run]`.
app.command(name="repair-hashes", help="Repair cache-key hashes after schema changes.")(repair.repair_hashes)


@app.callback()
def _main(
    root: Annotated[Path | None, typer.Option(
        "--root", help="Claudesync export tree (default: $CSINDEX_ROOT, then CWD).",
    )] = None,
    prompts_dir: Annotated[Path | None, typer.Option(
        "--prompts-dir",
        help="Directory of prompt-template overrides (per-file fallback to packaged defaults).",
    )] = None,
) -> None:
    # Records the requested root without validating it, so `--help` works
    # even outside an export tree. Each command validates on entry via
    # _require_root_or_exit() / paths.require_export_root().
    paths.set_requested_root(root)
    prompt_loader.set_prompts_dir(prompts_dir)


def _require_root_or_exit() -> Path:
    """Validate the resolved export root or exit DATAERR (65). Call first in every command."""
    try:
        return paths.require_export_root()
    except paths.InvalidExportTree as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exit_codes.DATAERR) from exc


def _preflight_claude_binary() -> None:
    """quick-mode only: it shells out to `claude -p` directly."""
    if shutil.which("claude") is not None:
        return
    print("csindex: missing required commands: claude", file=sys.stderr)
    print("    claude: https://claude.com/claude-code", file=sys.stderr)
    sys.exit(exit_codes.CONFIG)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _standalone_dirs(root: Path) -> list[Path]:
    d = root / "conversations"
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_dir())


def _nested_dirs(root: Path) -> list[Path]:
    d = root / "projects"
    if not d.is_dir():
        return []
    out: list[Path] = []
    for proj in sorted(p for p in d.iterdir() if p.is_dir()):
        c = proj / "conversations"
        if c.is_dir():
            out.extend(sorted(p for p in c.iterdir() if p.is_dir()))
    return out


def _project_dirs(root: Path) -> list[Path]:
    d = root / "projects"
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Step orchestration — prepare items, hand to runner.run_step
# ---------------------------------------------------------------------------

async def _run_leaves(
    provider: Provider, dirs: list[Path], *,
    batch_size: int, force: bool, scope: str,
    max_in_flight: int = 4, wait: bool = False,
) -> None:
    olog = log.get("orchestrator")
    if not dirs:
        olog.info("step_skipped", step="leaf", reason="no_targets", scope=scope)
        return
    olog.info("step_started", step="leaf", batch_size=batch_size, targets=len(dirs), scope=scope)

    # Prepare all items synchronously (fast: hash + file reads).
    items: list[workers.LeafItem] = []
    for d in dirs:
        item = workers.prepare_leaf(d, provider=provider, force=force)
        if item is not None:
            items.append(item)
    olog.info("step_prepared", step="leaf", to_process=len(items), cached_or_skipped=len(dirs) - len(items))

    await runner_mod.run_step(
        provider, step="leaf", items=items,
        schema_cls=models.LeafSummary,
        finalize_fn=workers.finalize_leaf,
        batch_size=batch_size,
        group_by_model=True,
        max_in_flight=max_in_flight,
        wait=wait,
    )
    _print_step_total("leaf")


async def _run_projects(
    provider: Provider, dirs: list[Path], *,
    batch_size: int, force: bool,
    max_in_flight: int = 4, wait: bool = False,
) -> None:
    olog = log.get("orchestrator")
    if not dirs:
        olog.info("step_skipped", step="project", reason="no_targets")
        return
    olog.info("step_started", step="project", batch_size=batch_size, targets=len(dirs))

    items: list[workers.ProjectItem] = []
    for d in dirs:
        item = workers.prepare_project(d, provider=provider, force=force)
        if item is not None:
            items.append(item)
    olog.info("step_prepared", step="project", to_process=len(items), cached_or_skipped=len(dirs) - len(items))

    await runner_mod.run_step(
        provider, step="project", items=items,
        schema_cls=models.ProjectAggregate,
        finalize_fn=workers.finalize_project,
        batch_size=batch_size,
        max_in_flight=max_in_flight,
        wait=wait,
    )
    _print_step_total("project")


async def _run_root(
    provider: Provider, *, batch_size: int, force: bool,
    max_in_flight: int = 4, wait: bool = False,
) -> None:
    olog = log.get("orchestrator")
    olog.info("step_started", step="root")
    item = workers.prepare_root(paths.EXPORT_ROOT, provider=provider, force=force)
    if item is None:
        _print_step_total("root")
        return

    # escalate=False preserves the pre-provider behavior: root already runs
    # the strongest configured model; "escalating" it to the (smaller)
    # escalation slot would be a downgrade.
    await runner_mod.run_step(
        provider, step="root", items=[item],
        schema_cls=models.RootAggregate,
        finalize_fn=workers.finalize_root,
        batch_size=batch_size,
        max_in_flight=max_in_flight,
        wait=wait,
        escalate=False,
    )
    _print_step_total("root")


def _print_step_total(step: str) -> None:
    agg = cost_log.aggregate(step=step)
    log.get("orchestrator").info("step_total", step=step, **agg)


def _print_grand_total() -> None:
    agg = cost_log.aggregate()
    log.get("orchestrator").info("grand_total", **agg)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _common_setup(
    log_level: str | None,
    log_format: str | None,
    no_color: bool,
    *,
    log_file: str | None = None,
    no_log_file: bool = False,
) -> None:
    load_dotenv(paths.EXPORT_ROOT / ".env")
    if no_color:
        os.environ["NO_COLOR"] = "1"
    # Default log file path next to export root, unless caller disabled.
    # Use direct assignment (not setdefault) so per-invocation paths.EXPORT_ROOT
    # propagates correctly across runs (e.g., tests with patched paths.EXPORT_ROOT).
    os.environ["CSINDEX_ROOT"] = str(paths.EXPORT_ROOT)
    log.configure(level=log_level, fmt=log_format, log_file=log_file, no_log_file=no_log_file)
    os.environ.setdefault("CSINDEX_COST_LOG", str(paths.EXPORT_ROOT / ".reindex-costs.jsonl"))
    os.environ.setdefault("CSINDEX_FAILURE_LOG", str(paths.EXPORT_ROOT / ".reindex-failures.jsonl"))
    cost_log.truncate()
    failures.truncate()
    log.get("orchestrator").info(
        "logs_init",
        cost_log=os.environ["CSINDEX_COST_LOG"],
        failure_log=os.environ["CSINDEX_FAILURE_LOG"],
        log_file=(
            "disabled" if no_log_file
            else (log_file or os.environ.get("CSINDEX_LOG_FILE") or str(paths.EXPORT_ROOT / ".reindex.log.jsonl"))
        ),
    )


def _resolve_provider_flags(
    provider_flag: config.ProviderName | None, use_api: bool, use_subscription: bool,
) -> config.ProviderName:
    """Map the flag triplet to a provider name. --provider wins; --api and
    --subscription are aliases kept for muscle memory."""
    if use_api and use_subscription:
        log.get("orchestrator").error("contradictory_flags", flag="--api --subscription")
        raise typer.Exit(exit_codes.USAGE)
    if provider_flag:
        return provider_flag
    if use_api:
        return config.ProviderName.ANTHROPIC
    if use_subscription:
        return config.ProviderName.CLAUDE_CLI
    return config.resolve_provider_name(paths.EXPORT_ROOT)


def _gather_leaves(scope: str, limit: int) -> list[Path]:
    standalone = _standalone_dirs(paths.EXPORT_ROOT) if scope in ("standalone", "both") else []
    nested = _nested_dirs(paths.EXPORT_ROOT) if scope in ("nested", "both") else []
    log.get("orchestrator").debug(
        "dir_scan", standalone=len(standalone), project_nested=len(nested), scope=scope,
    )
    dirs = standalone + nested
    if limit > 0:
        dirs = dirs[:limit]
        log.get("orchestrator").debug("dir_scan_limited", limit=limit, resulting=len(dirs))
    return dirs


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def quick(
    root: Annotated[Path | None, typer.Option(
        "--root", help="Claudesync export tree (default: $CSINDEX_ROOT, then CWD). "
        "Also accepted before the subcommand: `csindex --root PATH quick ...`.",
    )] = None,
    log_file: Annotated[str | None, typer.Option("--log-file")] = None,
    no_log_file: Annotated[bool, typer.Option("--no-log-file")] = False,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    log_format: Annotated[str | None, typer.Option("--log-format")] = None,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
    json_logs: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", "--debug")] = False,
) -> None:
    """Counts + top README/METADATA only (cheap, subscription-only)."""
    if root is not None:
        paths.set_requested_root(root)
    _require_root_or_exit()

    if json_logs:
        log_format = "json"
    if quiet:
        log_level = "warn"
    if verbose:
        log_level = "debug"
    _common_setup(log_level, log_format, no_color,
                  log_file=log_file, no_log_file=no_log_file)
    _preflight_claude_binary()

    olog = log.get("orchestrator")
    prompt = prompt_loader.load_prompt("quick").replace("{{EXPORT_DIR}}", str(paths.EXPORT_ROOT))
    try:
        with single_instance(paths.EXPORT_ROOT):
            proc = subprocess.run(
                [
                    "claude", "-p",
                    "--permission-mode", "acceptEdits",
                    "--add-dir", str(paths.EXPORT_ROOT),
                    "--output-format", "json",
                ],
                input=prompt,
                text=True,
            )
    except RuntimeError as e:
        olog.error("lock_failed", error=str(e))
        raise typer.Exit(exit_codes.TEMPFAIL) from e
    if proc.returncode != 0:
        raise typer.Exit(exit_codes.UNAVAILABLE)
    raise typer.Exit(exit_codes.OK)


@app.command()
def full(
    root: Annotated[Path | None, typer.Option(
        "--root", help="Claudesync export tree (default: $CSINDEX_ROOT, then CWD). "
        "Also accepted before the subcommand: `csindex --root PATH full ...`.",
    )] = None,
    only_leaves: Annotated[bool, typer.Option("--only-leaves")] = False,
    only_projects: Annotated[bool, typer.Option("--only-projects")] = False,
    only_root: Annotated[bool, typer.Option("--only-root")] = False,
    no_projects: Annotated[bool, typer.Option("--no-projects")] = False,
    limit: Annotated[int, typer.Option("--limit")] = 0,
    force: Annotated[bool, typer.Option("--force")] = False,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 100,
    provider_flag: Annotated[config.ProviderName | None, typer.Option(
        "--provider", help="AI provider. Default comes from $CSINDEX_PROVIDER / reindex.toml / claude-cli.",
    )] = None,
    use_api: Annotated[bool, typer.Option("--api", help="Alias for --provider anthropic.")] = False,
    use_subscription: Annotated[bool, typer.Option("--subscription", help="Alias for --provider claude-cli.")] = False,
    max_in_flight: Annotated[int, typer.Option(
        "--max-in-flight", help="Max concurrent batches in flight (batch-capable providers only). Default 4.",
    )] = 4,
    wait: Annotated[bool, typer.Option(
        "--wait",
        help=(
            "(batch-capable providers only) Block until batches complete + "
            "finalize results. Default: submit-and-exit."
        ),
    )] = False,
    log_file: Annotated[str | None, typer.Option(
        "--log-file", help="JSONL log file path. Default: <export>/.reindex.log.jsonl",
    )] = None,
    no_log_file: Annotated[bool, typer.Option("--no-log-file", help="Disable file logging entirely.")] = False,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    log_format: Annotated[str | None, typer.Option("--log-format")] = None,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
    json_logs: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", "--debug")] = False,
) -> None:
    """Depth-first hierarchical index: leaves -> projects -> root.

    --batch-size sets the batch chunk size on batch-capable providers, OR
    the per-item invoke concurrency on everything else. Default 100.

    --max-in-flight bounds concurrent in-flight batches on batch-capable
    providers; passing it with a non-batch provider is rejected.
    """
    # `--root` is accepted both before the subcommand (parsed by the app
    # callback into paths._requested_cli_root) and here, after `full`; a
    # value given here wins since it's the more specific placement.
    if root is not None:
        paths.set_requested_root(root)
    _require_root_or_exit()

    if json_logs:
        log_format = "json"
    if quiet:
        log_level = "warn"
    if verbose:
        log_level = "debug"

    provider_name = _resolve_provider_flags(provider_flag, use_api, use_subscription)
    batch_capable = issubclass(provider_class(provider_name), BatchCapable)

    # Reject --max-in-flight + --wait on providers with no batch mode.
    if not batch_capable and (max_in_flight != 4 or wait):
        log.configure(level=log_level, fmt=log_format)
        log.get("orchestrator").error(
            "batch_only_flag_without_batch_provider",
            provider=provider_name,
            offending_flags=[
                f for f, set_ in [("--max-in-flight", max_in_flight != 4), ("--wait", wait)] if set_
            ],
        )
        raise typer.Exit(exit_codes.USAGE)

    _common_setup(log_level, log_format, no_color,
                  log_file=log_file, no_log_file=no_log_file)
    olog = log.get("orchestrator")

    # Construct AFTER dotenv so api-key preflights see .env values.
    provider = get_provider(provider_name, config.load(paths.EXPORT_ROOT, provider_name=provider_name))

    only_set = sum([only_leaves, only_projects, only_root])
    if only_set > 1:
        olog.error("multiple_only_flags")
        raise typer.Exit(exit_codes.USAGE)

    if limit > 0 and only_set == 0:
        olog.warn("limit_gated", reason="aggregates need full leaves", effective_step="leaves")
        only_leaves = True

    leaf_scope = "standalone" if no_projects else "both"

    async def _orchestrate() -> None:
        shutdown.install(asyncio.get_running_loop())

        try:
            try:
                await provider.preflight()
            except ProviderFailure as e:
                olog.error("provider_preflight_failed", provider=provider.name, error=str(e))
                raise typer.Exit(exit_codes.CONFIG) from e

            # Drain any pending batches from a previous interrupted run.
            if isinstance(provider, BatchCapable):
                state = BatchState(paths.EXPORT_ROOT)
                if not state.is_empty():
                    await provider.resume_pending(
                        state=state,
                        schema_for_step=lambda step: models.STEP_MODEL[step],
                        finalize_persisted=workers.finalize_persisted,
                        max_in_flight=max_in_flight,
                    )

            if only_leaves:
                dirs = _gather_leaves(leaf_scope, limit)
                await _run_leaves(provider, dirs, batch_size=batch_size, force=force, scope=leaf_scope,
                                  max_in_flight=max_in_flight, wait=wait)
            elif only_projects:
                if no_projects:
                    olog.warn("contradictory_flags", flag="--no-projects --only-projects")
                    raise typer.Exit(exit_codes.USAGE)
                dirs = _gather_leaves("nested", limit)
                await _run_leaves(provider, dirs, batch_size=batch_size, force=force, scope="nested",
                                  max_in_flight=max_in_flight, wait=wait)
                await _run_projects(provider, _project_dirs(paths.EXPORT_ROOT), batch_size=batch_size, force=force,
                                    max_in_flight=max_in_flight, wait=wait)
            elif only_root:
                await _run_root(provider, batch_size=batch_size, force=force,
                                max_in_flight=max_in_flight, wait=wait)
            else:
                dirs = _gather_leaves(leaf_scope, limit)
                await _run_leaves(provider, dirs, batch_size=batch_size, force=force, scope=leaf_scope,
                                  max_in_flight=max_in_flight, wait=wait)
                if no_projects:
                    olog.info("step_skipped", step="project", reason="no_projects_flag")
                else:
                    await _run_projects(provider, _project_dirs(paths.EXPORT_ROOT), batch_size=batch_size, force=force,
                                        max_in_flight=max_in_flight, wait=wait)
                await _run_root(provider, batch_size=batch_size, force=force,
                                max_in_flight=max_in_flight, wait=wait)
        finally:
            shutdown.uninstall()
            await provider.aclose()

    exit_code = exit_codes.OK
    try:
        with single_instance(paths.EXPORT_ROOT):
            try:
                asyncio.run(_orchestrate())
            except typer.Exit as e:
                exit_code = e.exit_code
            except asyncio.CancelledError:
                # Graceful SIGINT path — shutdown handler cancelled tasks.
                olog.warn("shutdown_complete", reason="sigint")
                exit_code = 130
            except Exception as e:
                olog.error("unhandled", error=str(e)[:500])
                exit_code = exit_codes.SOFTWARE
    except RuntimeError as e:
        olog.error("lock_failed", error=str(e))
        exit_code = exit_codes.TEMPFAIL
    finally:
        _print_grand_total()
        fail_n = failures.count()
        if fail_n > 0 and exit_code == exit_codes.OK:
            exit_code = exit_codes.TEMPFAIL
        log.get("orchestrator").info("pipeline_done", failures=fail_n, exit=exit_code)
    raise typer.Exit(exit_code)


# ---------------------------------------------------------------------------
# Embedding (optional extra: uv sync --extra embed)
# ---------------------------------------------------------------------------

@app.command()
def embed(
    root: Annotated[Path | None, typer.Option(
        "--root", help="Claudesync export tree (default: $CSINDEX_ROOT, then CWD). "
        "Also accepted before the subcommand: `csindex --root PATH embed ...`.",
    )] = None,
    limit: Annotated[int, typer.Option("--limit", help="Cap number of conversations (0 = all).")] = 0,
    backend: Annotated[str | None, typer.Option(
        "--backend", help="Embedding backend: cloudflare, ollama, or openai. "
        "Default from $CSINDEX_EMBED_BACKEND / reindex.toml [embedding].backend.",
    )] = None,
    model: Annotated[str | None, typer.Option(
        "--model", help="Embedding model. Default: backend-specific default when omitted.",
    )] = None,
    base_url: Annotated[str | None, typer.Option(
        "--base-url", help="Base URL for the openai/ollama backends. "
        "Default from $CSINDEX_EMBED_BASE_URL / reindex.toml [embedding].base_url.",
    )] = None,
    chunk_chars: Annotated[int, typer.Option("--chunk-chars")] = 2000,
    overlap: Annotated[int, typer.Option("--overlap")] = 200,
    force: Annotated[bool, typer.Option(
        "--force", help="Re-embed all files, ignoring the content-hash cache.",
    )] = False,
    max_in_flight: Annotated[int, typer.Option(
        "--max-in-flight", help="Concurrent files embedding at once. Default 8.",
    )] = 8,
    no_conversations: Annotated[bool, typer.Option(
        "--no-conversations", help="Skip raw conversation.md sources.",
    )] = False,
    no_summaries: Annotated[bool, typer.Option(
        "--no-summaries", help="Skip generated INDEX.md summary sources.",
    )] = False,
    persist: Annotated[str | None, typer.Option("--persist", help="Chroma dir. Default <export>/.vector-db")] = None,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
    json_logs: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", "--debug")] = False,
) -> None:
    """EXPERIMENT: embed full conversation.md + INDEX.md summary contents into a
    local vector DB (Cloudflare bge-m3 -> Chroma). Orthogonal to the index.md
    hierarchy. Tag each chunk kind=conversation|summary for filtered search."""
    if root is not None:
        paths.set_requested_root(root)
    _require_root_or_exit()

    from reindex import embedding

    level = "warn" if quiet else ("debug" if verbose else None)
    _common_setup(level, "json" if json_logs else None, no_color)
    olog = log.get("orchestrator")
    persist_dir = Path(persist) if persist else paths.EXPORT_ROOT / ".vector-db"
    if no_conversations and no_summaries:
        olog.error("embed_nothing_to_do", reason="--no-conversations and --no-summaries")
        raise typer.Exit(exit_codes.USAGE)
    try:
        backend_r, model_r, base_url_r = config.resolve_embedding(backend, model, base_url, paths.EXPORT_ROOT)
        embedder = embedding.make_embedder(backend_r, model=model_r, base_url=base_url_r)
    except embedding.EmbeddingConfigError as e:
        olog.error("embed_config", error=str(e))
        raise typer.Exit(exit_codes.CONFIG) from e

    async def _orchestrate() -> embedding.EmbedStats:
        shutdown.install(asyncio.get_running_loop())
        try:
            return await embedding.embed_corpus(
                paths.EXPORT_ROOT, persist_dir, embedder=embedder,
                backend=backend_r, model=embedder.model,
                limit=limit, max_chars=chunk_chars, overlap=overlap,
                force=force, max_in_flight=max_in_flight,
                conversations=not no_conversations, summaries=not no_summaries,
            )
        finally:
            shutdown.uninstall()
            await embedder.aclose()

    exit_code = exit_codes.OK
    try:
        with single_instance(paths.EXPORT_ROOT):
            try:
                stats = asyncio.run(_orchestrate())
                olog.info("embed_complete", files=stats.files, chunks=stats.chunks,
                          skipped=stats.skipped, failed=stats.failed)
                if stats.failed > 0:
                    exit_code = exit_codes.TEMPFAIL  # cron retries the stragglers
            except asyncio.CancelledError:
                olog.warn("shutdown_complete", reason="sigint")
                exit_code = 130
            except embedding.CollectionMismatch as e:
                olog.error("embed_collection_mismatch", error=str(e))
                exit_code = exit_codes.DATAERR
            except RuntimeError as e:
                olog.error("embed_failed", error=str(e)[:500])
                exit_code = exit_codes.SOFTWARE
    except RuntimeError as e:
        olog.error("lock_failed", error=str(e))
        exit_code = exit_codes.TEMPFAIL
    raise typer.Exit(exit_code)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Natural-language query.")],
    root: Annotated[Path | None, typer.Option(
        "--root", help="Claudesync export tree (default: $CSINDEX_ROOT, then CWD). "
        "Also accepted before the subcommand: `csindex --root PATH search ...`.",
    )] = None,
    k: Annotated[int, typer.Option("--k", help="Number of results.")] = 5,
    kind: Annotated[str | None, typer.Option(
        "--kind", help="Filter to 'conversation' or 'summary'. Default: both.",
    )] = None,
    backend: Annotated[str | None, typer.Option(
        "--backend", help="Embedding backend: cloudflare, ollama, or openai. "
        "Default from $CSINDEX_EMBED_BACKEND / reindex.toml [embedding].backend.",
    )] = None,
    model: Annotated[str | None, typer.Option(
        "--model", help="Embedding model. Default: backend-specific default when omitted.",
    )] = None,
    base_url: Annotated[str | None, typer.Option(
        "--base-url", help="Base URL for the openai/ollama backends. "
        "Default from $CSINDEX_EMBED_BASE_URL / reindex.toml [embedding].base_url.",
    )] = None,
    persist: Annotated[str | None, typer.Option("--persist")] = None,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", "--debug")] = False,
) -> None:
    """EXPERIMENT: semantic search over the embedded corpus."""
    if root is not None:
        paths.set_requested_root(root)
    _require_root_or_exit()

    from reindex import embedding

    _common_setup("debug" if verbose else "warn", None, no_color, no_log_file=True)
    persist_dir = Path(persist) if persist else paths.EXPORT_ROOT / ".vector-db"
    try:
        backend_r, model_r, base_url_r = config.resolve_embedding(backend, model, base_url, paths.EXPORT_ROOT)
        embedder = embedding.make_embedder(backend_r, model=model_r, base_url=base_url_r)
    except embedding.EmbeddingConfigError as e:
        log.get("orchestrator").error("search_config", error=str(e))
        raise typer.Exit(exit_codes.CONFIG) from e

    async def _run() -> list[dict]:
        try:
            return await embedding.search(
                persist_dir, query, embedder=embedder,
                backend=backend_r, model=embedder.model, k=k, kind=kind,
            )
        finally:
            await embedder.aclose()

    try:
        hits = asyncio.run(_run())
    except embedding.CollectionMismatch as e:
        log.get("orchestrator").error("search_collection_mismatch", error=str(e))
        raise typer.Exit(exit_codes.DATAERR) from e

    for h in hits:
        snippet = " ".join(h["text"].split())[:200]
        print(f"\n[{h['score']:.3f}] ({h['kind']}) {h['slug']}\n  {snippet}")


@app.command(name="embed-migrate")
def embed_migrate(
    root: Annotated[Path | None, typer.Option(
        "--root", help="Claudesync export tree (default: $CSINDEX_ROOT, then CWD). "
        "Also accepted before the subcommand: `csindex --root PATH embed-migrate ...`.",
    )] = None,
    persist: Annotated[str | None, typer.Option("--persist")] = None,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
) -> None:
    """One-shot: tag pre-`kind` entries as kind=conversation (metadata-only,
    reuses stored vectors — no re-embedding). Run once before the first
    summary-embed so old data filters correctly. Idempotent."""
    if root is not None:
        paths.set_requested_root(root)
    _require_root_or_exit()

    from reindex import embedding

    _common_setup(None, None, no_color, no_log_file=True)
    persist_dir = Path(persist) if persist else paths.EXPORT_ROOT / ".vector-db"
    n = embedding.backfill_kind(persist_dir)
    log.get("orchestrator").info("embed_migrate_done", chunks_tagged=n)
    raise typer.Exit(exit_codes.OK)


if __name__ == "__main__":
    app()
