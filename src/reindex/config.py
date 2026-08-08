"""
Provider configuration: model tiers, pricing, selection.

Defaults baked into _BUILTIN_DEFAULTS reproduce the pre-refactor hardcoded
behavior exactly — with no `reindex.toml` present, nothing changes.

Precedence for provider selection:
  CLI flag > $CSINDEX_PROVIDER > reindex.toml [reindex].provider > "claude-cli"

$CSINDEX_ROOT_MODEL overrides the root-tier model for any provider,
independent of the [reindex.providers.*.models] TOML section.

TOML shape (optional file at EXPORT_ROOT/reindex.toml):

    [reindex]
    provider = "claude-cli"

    [reindex.providers.ollama]
    base_url = "http://127.0.0.1:11434"          # provider options, inline

    [reindex.providers.ollama.models]
    leaf = "llama3.2"
    project = "llama3.2"
    root = "llama3.2"

    [reindex.providers.anthropic.pricing]
    "claude-haiku-4-5" = { input = 1.0, output = 5.0 }
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from reindex import log


class ProviderName(StrEnum):
    """Closed set of provider names. The registry is a code-level dict —
    adding a provider means adding code — so the name space is honest as
    an enum: typer gets --provider choices/validation for free and the
    string literals stop being stringly-typed."""

    CLAUDE_CLI = "claude-cli"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OPENCODE = "opencode"


DEFAULT_PROVIDER = ProviderName.CLAUDE_CLI

# USD per 1M tokens: prefix -> (input, output). Prefix-matched so dated
# variants (claude-haiku-4-5-20251001) hit the base tier. The [1m] long-
# context premium above 200K input is NOT modeled; escalation retries
# rarely cross it.
_ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7": (15.0, 75.0),
}


@dataclass(frozen=True)
class ModelTiers:
    """Which model summarizes each step, plus the escalation slot.

    leaf_large kicks in for transcripts over large_leaf_kb (pre-refactor
    `_pick_leaf_model`: >600KB went to Sonnet instead of Haiku).
    escalation=None means the runner never retries on a stronger model.
    """

    leaf: str
    project: str
    root: str
    leaf_large: str | None = None
    large_leaf_kb: int = 600
    escalation: str | None = None

    def for_step(self, step: str, *, size_bytes: int = 0) -> str:
        if step == "leaf" and self.leaf_large and size_bytes > self.large_leaf_kb * 1024:
            return self.leaf_large
        return {"leaf": self.leaf, "project": self.project, "root": self.root}[step]


@dataclass
class ProviderConfig:
    name: str
    models: ModelTiers
    pricing: dict[str, tuple[float, float]] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    # Models already warned about this run; an empty pricing table means a
    # deliberately free provider (Ollama) and never warns.
    _warned_models: set[str] = field(default_factory=set, repr=False)

    def compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """USD cost for one call. Warns once per unknown model instead of
        silently returning $0 (which undercounted e.g. unlisted tiers)."""
        for prefix, (in_rate, out_rate) in self.pricing.items():
            if model.startswith(prefix):
                return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
        if self.pricing and model not in self._warned_models:
            self._warned_models.add(model)
            log.get("cost").warning("unknown_model_pricing", model=model, provider=self.name)
        return 0.0


# Sonnet 4.6 [1m] for the escalation slot: the cases that need escalation
# are role-lock losses on multi-turn tool-use (reasoning ceiling), and the
# file-mode Read-loop on >200KB transcripts, which can thrash autocompact
# in a 200K context window on long, tool-heavy runs. 1M context fits the
# whole file. Same per-token price as standard below 200K.
_ANTHROPIC_TIERS = ModelTiers(
    leaf="claude-haiku-4-5-20251001",
    leaf_large="claude-sonnet-4-6",
    project="claude-haiku-4-5-20251001",
    root="claude-opus-4-7",
    escalation="claude-sonnet-4-6[1m]",
)

_BUILTIN_DEFAULTS: dict[ProviderName, dict[str, Any]] = {
    ProviderName.CLAUDE_CLI: {"models": _ANTHROPIC_TIERS, "pricing": dict(_ANTHROPIC_PRICING), "options": {}},
    ProviderName.ANTHROPIC: {"models": _ANTHROPIC_TIERS, "pricing": dict(_ANTHROPIC_PRICING), "options": {}},
    ProviderName.OLLAMA: {
        "models": ModelTiers(leaf="llama3.2", project="llama3.2", root="llama3.2"),
        "pricing": {},  # free — empty table never warns
        "options": {"base_url": "http://127.0.0.1:11434"},
    },
    ProviderName.OPENCODE: {
        "models": ModelTiers(
            leaf="google/gemini-2.5-flash",
            project="google/gemini-2.5-flash",
            root="google/gemini-2.5-flash",
        ),
        # Gemini 2.5 Flash list price (paid tier); free-tier usage costs $0
        # but logging list price keeps the "what would this cost" signal.
        "pricing": {"google/gemini-2.5-flash": (0.30, 2.50)},
        "options": {},
    },
}


def coerce_provider_name(value: str | ProviderName) -> ProviderName:
    try:
        return ProviderName(value)
    except ValueError as e:
        raise SystemExit(
            f"unknown provider {value!r}; known: {[p.value for p in ProviderName]}"
        ) from e


def _read_toml_raw(export_root: Path) -> dict[str, Any]:
    """Parse `reindex.toml` at the export root (or `{}` if absent), full
    top-level table. Callers slice into whichever section they need — the
    `[reindex]` table (`_read_toml`) or the top-level `[embedding]` table
    (`resolve_embedding`) — without re-parsing the file."""
    f = export_root / "reindex.toml"
    if not f.is_file():
        return {}
    with f.open("rb") as fh:
        return tomllib.load(fh)


def _read_toml(export_root: Path) -> dict[str, Any]:
    return _read_toml_raw(export_root).get("reindex", {})


def resolve_provider_name(
    export_root: Path, cli_value: str | ProviderName | None = None,
) -> ProviderName:
    if cli_value:
        return coerce_provider_name(cli_value)
    env = os.environ.get("CSINDEX_PROVIDER")
    if env:
        return coerce_provider_name(env)
    return coerce_provider_name(_read_toml(export_root).get("provider", DEFAULT_PROVIDER))


def _tiers_from_toml(base: ModelTiers, section: dict[str, Any]) -> ModelTiers:
    return ModelTiers(
        leaf=section.get("leaf", base.leaf),
        project=section.get("project", base.project),
        root=section.get("root", base.root),
        leaf_large=section.get("leaf_large", base.leaf_large),
        large_leaf_kb=section.get("large_leaf_kb", base.large_leaf_kb),
        escalation=section.get("escalation", base.escalation),
    )


def load(export_root: Path, *, provider_name: str | ProviderName | None = None) -> ProviderConfig:
    """Build the effective ProviderConfig for `provider_name` (or the
    resolved default)."""
    name = coerce_provider_name(provider_name) if provider_name else resolve_provider_name(export_root)
    defaults = _BUILTIN_DEFAULTS[name]

    toml_section: dict[str, Any] = _read_toml(export_root).get("providers", {}).get(name, {})
    models_section = toml_section.get("models", {})
    pricing_section = toml_section.get("pricing", {})

    tiers = _tiers_from_toml(defaults["models"], models_section)

    # $CSINDEX_ROOT_MODEL overrides the root-tier model for any provider.
    root_env = os.environ.get("CSINDEX_ROOT_MODEL")
    if root_env:
        tiers = ModelTiers(
            leaf=tiers.leaf, project=tiers.project, root=root_env,
            leaf_large=tiers.leaf_large, large_leaf_kb=tiers.large_leaf_kb,
            escalation=tiers.escalation,
        )

    pricing = dict(defaults["pricing"])
    for prefix, rates in pricing_section.items():
        pricing[prefix] = (float(rates["input"]), float(rates["output"]))

    options = dict(defaults["options"])
    options.update({k: v for k, v in toml_section.items() if k not in ("models", "pricing")})

    return ProviderConfig(name=name, models=tiers, pricing=pricing, options=options)


def resolve_embedding(
    cli_backend: str | None,
    cli_model: str | None,
    cli_base_url: str | None,
    root: Path,
) -> tuple[str, str | None, str | None]:
    """Backend/model/base_url for the embed/search commands, precedence
    CLI flag > $CSINDEX_EMBED_* > top-level `[embedding]` table in
    reindex.toml. Raises embedding.EmbeddingConfigError when no backend is
    configured anywhere."""
    from reindex import embedding as _e

    toml_cfg = _read_toml_raw(root).get("embedding", {})
    backend = cli_backend or os.environ.get("CSINDEX_EMBED_BACKEND") or toml_cfg.get("backend")
    model = cli_model or os.environ.get("CSINDEX_EMBED_MODEL") or toml_cfg.get("model")
    base_url = cli_base_url or os.environ.get("CSINDEX_EMBED_BASE_URL") or toml_cfg.get("base_url")
    if not backend:
        raise _e.EmbeddingConfigError(
            "No embedding backend configured. Set one of: --backend flag, "
            "$CSINDEX_EMBED_BACKEND, or [embedding].backend in reindex.toml. "
            "Backends: cloudflare, ollama, openai."
        )
    return backend, model, base_url
