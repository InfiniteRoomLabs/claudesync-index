"""config.py — provider selection, model tiers, pricing, TOML overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from reindex import config

# ---------------------------------------------------------------------------
# Defaults (no reindex.toml) must reproduce pre-refactor hardcoded behavior
# ---------------------------------------------------------------------------

def test_default_provider_is_claude_cli(tmp_path: Path):
    assert config.resolve_provider_name(tmp_path) == "claude-cli"


def test_claude_cli_default_tiers_match_legacy(tmp_path: Path):
    cfg = config.load(tmp_path, provider_name="claude-cli")
    assert cfg.models.leaf == "claude-haiku-4-5-20251001"
    assert cfg.models.project == "claude-haiku-4-5-20251001"
    assert cfg.models.root == "claude-opus-4-7"
    assert cfg.models.escalation == "claude-sonnet-4-6[1m]"


def test_leaf_large_threshold_matches_legacy_pick(tmp_path: Path):
    """Pre-refactor _pick_leaf_model: >600KB -> Sonnet, else dated Haiku."""
    cfg = config.load(tmp_path, provider_name="claude-cli")
    assert cfg.models.for_step("leaf", size_bytes=600 * 1024) == "claude-haiku-4-5-20251001"
    assert cfg.models.for_step("leaf", size_bytes=600 * 1024 + 1) == "claude-sonnet-4-6"


def test_for_step_project_and_root(tmp_path: Path):
    cfg = config.load(tmp_path, provider_name="claude-cli")
    assert cfg.models.for_step("project") == "claude-haiku-4-5-20251001"
    assert cfg.models.for_step("root") == "claude-opus-4-7"


def test_for_step_unknown_raises(tmp_path: Path):
    cfg = config.load(tmp_path, provider_name="claude-cli")
    with pytest.raises(KeyError):
        cfg.models.for_step("nonsense")


def test_root_model_env_backcompat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CSINDEX_ROOT_MODEL", "claude-sonnet-4-6")
    cfg = config.load(tmp_path, provider_name="claude-cli")
    assert cfg.models.root == "claude-sonnet-4-6"
    # other tiers untouched
    assert cfg.models.leaf == "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Provider selection precedence
# ---------------------------------------------------------------------------

def test_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "reindex.toml").write_text('[reindex]\nprovider = "ollama"\n', encoding="utf-8")
    monkeypatch.setenv("CSINDEX_PROVIDER", "anthropic")
    assert config.resolve_provider_name(tmp_path) == "anthropic"


def test_cli_overrides_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CSINDEX_PROVIDER", "anthropic")
    assert config.resolve_provider_name(tmp_path, cli_value="ollama") == "ollama"


def test_toml_provider_used_when_no_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CSINDEX_PROVIDER", raising=False)
    (tmp_path / "reindex.toml").write_text('[reindex]\nprovider = "ollama"\n', encoding="utf-8")
    assert config.resolve_provider_name(tmp_path) == "ollama"


# ---------------------------------------------------------------------------
# TOML overrides
# ---------------------------------------------------------------------------

TOML = """
[reindex]
provider = "ollama"

[reindex.providers.ollama]
base_url = "http://ollama.lab:11434"

[reindex.providers.ollama.models]
leaf = "llama3.2"
root = "llama3.3:70b"

[reindex.providers.anthropic.pricing]
"claude-haiku-4-5" = { input = 2.0, output = 10.0 }
"""


def test_toml_models_merge_over_defaults(tmp_path: Path):
    (tmp_path / "reindex.toml").write_text(TOML, encoding="utf-8")
    cfg = config.load(tmp_path, provider_name="ollama")
    assert cfg.models.leaf == "llama3.2"
    assert cfg.models.root == "llama3.3:70b"
    assert cfg.models.project == "llama3.2"  # default survives partial override
    assert cfg.options["base_url"] == "http://ollama.lab:11434"


def test_toml_pricing_overrides_default(tmp_path: Path):
    (tmp_path / "reindex.toml").write_text(TOML, encoding="utf-8")
    cfg = config.load(tmp_path, provider_name="anthropic")
    assert cfg.pricing["claude-haiku-4-5"] == (2.0, 10.0)
    # untouched entries survive
    assert cfg.pricing["claude-opus-4-7"] == (15.0, 75.0)


def test_unknown_provider_name_rejected(tmp_path: Path):
    """Provider names are a closed StrEnum — unknown names fail loudly
    everywhere (CLI flag, env var, toml) instead of half-working."""
    with pytest.raises(SystemExit):
        config.load(tmp_path, provider_name="my-custom")
    with pytest.raises(SystemExit):
        config.coerce_provider_name("my-custom")


def test_provider_name_enum_accepts_string_values(tmp_path: Path):
    assert config.coerce_provider_name("ollama") is config.ProviderName.OLLAMA
    cfg = config.load(tmp_path, provider_name=config.ProviderName.OLLAMA)
    assert cfg.name == "ollama"


# ---------------------------------------------------------------------------
# compute_cost: warn-once semantics
# ---------------------------------------------------------------------------

def test_compute_cost_known_prefix(tmp_path: Path):
    cfg = config.load(tmp_path, provider_name="anthropic")
    assert cfg.compute_cost("claude-haiku-4-5-20251001", 1_000_000, 0) == pytest.approx(1.0)


def test_compute_cost_unknown_model_warns_once(tmp_path: Path):
    cfg = config.load(tmp_path, provider_name="anthropic")
    assert cfg.compute_cost("not-a-real-model", 1000, 1000) == 0.0
    assert "not-a-real-model" in cfg._warned_models
    # second call: still zero, no duplicate bookkeeping blowup
    assert cfg.compute_cost("not-a-real-model", 1000, 1000) == 0.0
    assert len(cfg._warned_models) == 1


def test_compute_cost_empty_pricing_is_silent_free(tmp_path: Path):
    cfg = config.load(tmp_path, provider_name="ollama")
    assert cfg.compute_cost("llama3.2", 1_000_000, 1_000_000) == 0.0
    assert cfg._warned_models == set()
