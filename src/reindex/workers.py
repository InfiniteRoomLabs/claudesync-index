"""
Step workers: leaf, project, root.

Each step is split into:
  prepare_X(dir, provider=..., force=...) -> WorkItem | None  (None == cache hit / skip)
  finalize_X(item, payload, cost_info)                        (validate already done by caller)

runner.run_step consumes WorkItems: BatchCapable providers get them as a
batch submission with finalize_X as the per-result callback; everyone else
gets a bounded per-item invoke loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from reindex import cost_log, hashing, log, paths, prompt_loader, render, stamp
from reindex.hashing import read_frontmatter_field
from reindex.providers.base import Provider


def _now_iso_z() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_template(name: str, **subs: str) -> str:
    text = prompt_loader.load_prompt(name)
    for k, v in subs.items():
        text = text.replace(f"{{{{{k}}}}}", v)
    return text


_fallback_provider: Provider | None = None


def _default_provider() -> Provider:
    """claude-cli with built-in config — what every prepare_X call meant
    before providers existed. Real runs pass the CLI-selected provider."""
    global _fallback_provider
    if _fallback_provider is None:
        from reindex import config
        from reindex.providers.claude_cli import ClaudeCliProvider

        _fallback_provider = ClaudeCliProvider(
            config.load(paths.EXPORT_ROOT, provider_name=config.ProviderName.CLAUDE_CLI)
        )
    return _fallback_provider


class TranscriptMode(StrEnum):
    """How the leaf transcript is delivered to the summarizing model.

    INLINE: full transcript text embedded in the user_content payload.
        Fast path; works up to the subscription per-message-bytes cap.
    FILE:   transcript stays on disk; the prompt instructs the model to
        call the Read tool on conversation.md. Subscription path must
        whitelist conv_dir via --add-dir for this mode to work.
    """

    INLINE = "inline"
    FILE = "file"


# Empirically derived from a full reindex run on 2026-05-14: every transcript
# >= ~226KB returned 'Prompt is too long' (api_error_status 400) when
# inlined into the claude -p stdin payload, while transcripts under that
# size succeeded. 200KB gives a 10% safety margin and rounds nicely.
#
# Transcripts above the threshold are summarized via a file-mode prompt that
# instructs the model to call the Read tool on conversation.md instead of
# receiving the transcript inline. The hypothesis is that the subscription
# path caps per-user-message bytes separately from total-context tokens, so
# Read-tool tool results don't share the same ceiling. If that turns out to
# be wrong we fall back to model-tier escalation.
_INLINE_TRANSCRIPT_THRESHOLD = 200 * 1024


def _pick_leaf_transcript_mode(size_bytes: int) -> TranscriptMode:
    if size_bytes <= _INLINE_TRANSCRIPT_THRESHOLD:
        return TranscriptMode.INLINE
    return TranscriptMode.FILE


# Escalation policy (retryable-kind judgment + escalation model) moved to
# providers: ProviderFailure.retryable is set by the provider, the
# escalation model lives in config.ModelTiers, and the retry loop is
# runner.invoke_with_escalation.


# ---------------------------------------------------------------------------
# WorkItem types — inputs + cache-key context for finalize
# ---------------------------------------------------------------------------

@dataclass
class LeafItem:
    slug: str
    conv_dir: Path
    out_file: Path
    content_hash: str
    model: str
    generated_at: str
    system_prompt: str
    user_content: str
    transcript_mode: TranscriptMode = TranscriptMode.INLINE


@dataclass
class ProjectItem:
    slug: str
    project_dir: Path
    out_file: Path
    children_hash: str
    model: str
    generated_at: str
    system_prompt: str
    user_content: str


@dataclass
class RootItem:
    out_file: Path
    inputs_hash: str
    model: str
    generated_at: str
    system_prompt: str
    user_content: str


# ---------------------------------------------------------------------------
# Leaf
# ---------------------------------------------------------------------------

def prepare_leaf(conv_dir: Path, *, provider: Provider | None = None, force: bool = False) -> LeafItem | None:
    """Compute hashes, check cache, build prompts. Returns None on cache hit / skip."""
    provider = provider or _default_provider()
    leaf_log = log.get("worker").bind(slug=conv_dir.name)

    conv_file = conv_dir / "conversation.md"
    if not conv_file.is_file():
        leaf_log.error("no_conversation_md")
        return None

    if conv_file.stat().st_size == 0:
        leaf_log.warn("skipped_empty")
        return None

    out_file = conv_dir / "INDEX.md"
    slug = conv_dir.name
    content_hash = hashing.leaf_hash(conv_file)
    size_bytes = conv_file.stat().st_size
    model_name = provider.model_for("leaf", size_bytes=size_bytes)
    # FILE mode needs the provider to actually have a Read tool; providers
    # without it always inline (pre-provider code picked FILE by size alone,
    # which on the batch path produced a prompt pointing at a file the API
    # could never read).
    if provider.supports_file_transcripts:
        transcript_mode = _pick_leaf_transcript_mode(size_bytes)
    else:
        transcript_mode = TranscriptMode.INLINE
    generated_at = _now_iso_z()
    leaf_log.debug(
        "hash_computed",
        content_hash=content_hash, size_bytes=size_bytes,
        model=model_name, transcript_mode=str(transcript_mode),
    )

    if not force and out_file.is_file():
        existing = read_frontmatter_field(out_file, "content_hash")
        if existing == content_hash:
            leaf_log.info("cache_hit")
            return None
        leaf_log.debug("cache_check", existing=existing, computed=content_hash)

    # Two prompt + payload shapes:
    #   INLINE: full transcript in user_content; prompt assumes it's inline.
    #   FILE:   user_content is a brief pointer; prompt instructs the model
    #           to Read conversation.md from disk. Subscription path
    #           whitelists conv_dir so Read works.
    if transcript_mode is TranscriptMode.INLINE:
        system_prompt = _read_template("conversation-summary", CONV_FILE=str(conv_file))
        user_content = conv_file.read_text(encoding="utf-8", errors="replace")
    else:
        system_prompt = _read_template("conversation-summary-file", CONV_FILE=str(conv_file))
        user_content = (
            f"The transcript for this conversation is at `{conv_file}`. "
            "Use the Read tool to load the file (no offset, no limit), then "
            "emit the JSON summary as instructed in the system prompt."
        )
    leaf_log.debug("prompt_rendered", system_bytes=len(system_prompt), user_bytes=len(user_content))

    return LeafItem(
        slug=slug,
        conv_dir=conv_dir,
        out_file=out_file,
        content_hash=content_hash,
        model=model_name,
        generated_at=generated_at,
        system_prompt=system_prompt,
        user_content=user_content,
        transcript_mode=transcript_mode,
    )


def finalize_leaf(
    item: LeafItem,
    payload: BaseModel,
    cost: float,
    turns: int,
    duration_ms: int,
    *,
    model: str | None = None,
    retry: bool = False,
) -> None:
    """Render validated payload, stamp cache-key fields, log cost.

    `model` overrides `item.model` for the stamp + cost record. When the
    escalation retry served the result, the model that actually produced
    it is the Sonnet escalation, not the Haiku item.model -- the stamped
    INDEX.md should reflect the producing model so future cache decisions
    and analytics see the truth.

    `retry` flags the cost record so analytics can tally escalation runs
    separately from primary runs.
    """
    used_model = model or item.model
    render.render_leaf(payload, item.out_file)  # type: ignore[arg-type]
    stamp.stamp_many(item.out_file, {
        "content_hash": item.content_hash,
        "slug": item.slug,
        "generated_at": item.generated_at,
        "model": used_model,
        "conversation_model": stamp.read_conversation_model(item.conv_dir),
    })
    cost_log.record(
        step="leaf", slug=item.slug,
        cost=cost, turns=turns, duration_ms=duration_ms,
        model=used_model, retry=retry,
    )
    log.get("worker").bind(slug=item.slug).info(
        "done", cost=cost, turns=turns, duration_ms=duration_ms,
        model=used_model, retry=retry,
    )


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

def prepare_project(project_dir: Path, *, provider: Provider | None = None, force: bool = False) -> ProjectItem | None:
    provider = provider or _default_provider()
    proj_log = log.get("worker").bind(slug=project_dir.name)
    out_file = project_dir / "INDEX.md"
    slug = project_dir.name
    children_hash = hashing.project_children_hash(project_dir)
    model_name = provider.model_for("project")
    generated_at = _now_iso_z()
    proj_log.debug("hash_computed", children_hash=children_hash)

    if not force and out_file.is_file():
        existing = read_frontmatter_field(out_file, "children_hash")
        if existing == children_hash:
            proj_log.info("cache_hit")
            return None

    payload_parts: list[str] = []
    convo_dir = project_dir / "conversations"
    if convo_dir.is_dir():
        for child_idx in sorted(convo_dir.glob("*/INDEX.md")):
            payload_parts.append(f"=== {child_idx.parent.name} ===\n{hashing.frontmatter_head(child_idx)}")
    kn_dir = project_dir / "knowledge"
    if kn_dir.is_dir():
        for kf in sorted(p for p in kn_dir.iterdir() if p.is_file()):
            payload_parts.append(f"=== knowledge/{kf.name} ===\n(file size: {kf.stat().st_size} bytes)")

    children_payload = "\n\n".join(payload_parts)
    system_prompt = _read_template("project-aggregate", PROJECT_DIR=str(project_dir))
    user_content = (
        f"Project: {slug}\n\n"
        "Child INDEX.md frontmatters and knowledge files follow:\n\n"
        f"{children_payload}"
    )

    return ProjectItem(
        slug=slug,
        project_dir=project_dir,
        out_file=out_file,
        children_hash=children_hash,
        model=model_name,
        generated_at=generated_at,
        system_prompt=system_prompt,
        user_content=user_content,
    )


def finalize_project(
    item: ProjectItem,
    payload: BaseModel,
    cost: float,
    turns: int,
    duration_ms: int,
    *,
    model: str | None = None,
    retry: bool = False,
) -> None:
    used_model = model or item.model
    render.render_project(payload, item.out_file)  # type: ignore[arg-type]
    stamp.stamp_many(item.out_file, {
        "children_hash": item.children_hash,
        "slug": item.slug,
        "generated_at": item.generated_at,
        "model": used_model,
    })
    cost_log.record(
        step="project", slug=item.slug,
        cost=cost, turns=turns, duration_ms=duration_ms,
        model=used_model, retry=retry,
    )
    log.get("worker").bind(slug=item.slug).info(
        "done", cost=cost, turns=turns, duration_ms=duration_ms,
        model=used_model, retry=retry,
    )


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

def prepare_root(export_dir: Path, *, provider: Provider | None = None, force: bool = False) -> RootItem | None:
    provider = provider or _default_provider()
    root_log = log.get("worker")
    out_file = export_dir / "INDEX.md"
    inputs_hash = hashing.root_inputs_hash(export_dir)
    model_name = provider.model_for("root")
    generated_at = _now_iso_z()
    root_log.debug("hash_computed", inputs_hash=inputs_hash)

    if not force and out_file.is_file():
        existing = read_frontmatter_field(out_file, "inputs_hash")
        if existing == inputs_hash:
            root_log.info("root_cache_hit")
            return None

    payload_parts: list[str] = []
    proj_dir = export_dir / "projects"
    if proj_dir.is_dir():
        for idx in sorted(proj_dir.glob("*/INDEX.md")):
            payload_parts.append(f"=== project/{idx.parent.name} ===\n{hashing.frontmatter_head(idx)}")
    convo_dir = export_dir / "conversations"
    if convo_dir.is_dir():
        for idx in sorted(convo_dir.glob("*/INDEX.md")):
            payload_parts.append(f"=== standalone/{idx.parent.name} ===\n{hashing.frontmatter_head(idx)}")

    children_payload = "\n\n".join(payload_parts)
    system_prompt = _read_template("root-aggregate", EXPORT_DIR=str(export_dir))
    user_content = "Project + standalone INDEX.md frontmatters follow:\n\n" + children_payload

    return RootItem(
        out_file=out_file,
        inputs_hash=inputs_hash,
        model=model_name,
        generated_at=generated_at,
        system_prompt=system_prompt,
        user_content=user_content,
    )


def finalize_root(
    item: RootItem,
    payload: BaseModel,
    cost: float,
    turns: int,
    duration_ms: int,
    *,
    model: str | None = None,
    retry: bool = False,
) -> None:
    """Same override semantics as the other finalizers so runner can call
    all three uniformly. Root never escalates today, so model/retry stay
    at their defaults in practice."""
    used_model = model or item.model
    render.render_root(payload, item.out_file)  # type: ignore[arg-type]
    stamp.stamp_many(item.out_file, {
        "inputs_hash": item.inputs_hash,
        "generated_at": item.generated_at,
        "model": used_model,
    })
    cost_log.record(
        step="root", slug="root",
        cost=cost, turns=turns, duration_ms=duration_ms, model=used_model,
        retry=retry,
    )
    log.get("worker").info("root_done", cost=cost, turns=turns, duration_ms=duration_ms)


# ---------------------------------------------------------------------------
# Persistence helpers (batch resumability)
# ---------------------------------------------------------------------------

def serialize_leaf_item(item: LeafItem) -> dict:
    """Reduce a LeafItem to JSON-serializable fields needed by finalize_leaf."""
    return {
        "out_file": str(item.out_file),
        "content_hash": item.content_hash,
        "slug": item.slug,
        "generated_at": item.generated_at,
        "model": item.model,
    }


def serialize_project_item(item: ProjectItem) -> dict:
    return {
        "out_file": str(item.out_file),
        "children_hash": item.children_hash,
        "slug": item.slug,
        "generated_at": item.generated_at,
        "model": item.model,
    }


def serialize_root_item(item: RootItem) -> dict:
    return {
        "out_file": str(item.out_file),
        "inputs_hash": item.inputs_hash,
        "generated_at": item.generated_at,
        "model": item.model,
    }


def _restore_leaf(d: dict) -> LeafItem:
    out_file = Path(d["out_file"])
    return LeafItem(
        slug=d["slug"], conv_dir=out_file.parent, out_file=out_file,
        content_hash=d["content_hash"], model=d["model"],
        generated_at=d["generated_at"],
        system_prompt="", user_content="",  # not needed for finalize
    )


def _restore_project(d: dict) -> ProjectItem:
    out_file = Path(d["out_file"])
    return ProjectItem(
        slug=d["slug"], project_dir=out_file.parent, out_file=out_file,
        children_hash=d["children_hash"], model=d["model"],
        generated_at=d["generated_at"],
        system_prompt="", user_content="",
    )


def _restore_root(d: dict) -> RootItem:
    return RootItem(
        out_file=Path(d["out_file"]),
        inputs_hash=d["inputs_hash"], model=d["model"],
        generated_at=d["generated_at"],
        system_prompt="", user_content="",
    )


async def finalize_persisted(
    step: str, step_kwargs: dict, payload: BaseModel,
    cost: float, turns: int, duration_ms: int,
) -> None:
    """Resume-time finalizer. Reconstructs the dataclass from persisted kwargs
    and calls the appropriate finalize_X."""
    if step == "leaf":
        finalize_leaf(_restore_leaf(step_kwargs), payload, cost, turns, duration_ms)
    elif step == "project":
        finalize_project(_restore_project(step_kwargs), payload, cost, turns, duration_ms)
    elif step == "root":
        finalize_root(_restore_root(step_kwargs), payload, cost, turns, duration_ms)
    else:
        raise ValueError(f"unknown step: {step}")
