"""config.py — provider selection, model tiers, pricing, TOML overrides."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

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


# ---------------------------------------------------------------------------
# resolve_embedding: backend/model/base_url precedence (CLI > env > toml)
# ---------------------------------------------------------------------------

def test_resolve_embedding_precedence(tmp_path, monkeypatch):
    (tmp_path / "reindex.toml").write_text(
        '[embedding]\nbackend = "cloudflare"\nmodel = "toml-model"\n', encoding="utf-8"
    )
    monkeypatch.setenv("CSINDEX_EMBED_BACKEND", "ollama")
    monkeypatch.setenv("CSINDEX_EMBED_MODEL", "env-model")
    from reindex import config
    # CLI beats env beats toml
    assert config.resolve_embedding("openai", "cli-model", None, tmp_path)[:2] == ("openai", "cli-model")
    assert config.resolve_embedding(None, None, None, tmp_path)[:2] == ("ollama", "env-model")
    monkeypatch.delenv("CSINDEX_EMBED_BACKEND")
    monkeypatch.delenv("CSINDEX_EMBED_MODEL")
    assert config.resolve_embedding(None, None, None, tmp_path)[:2] == ("cloudflare", "toml-model")


def test_resolve_embedding_unconfigured_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("CSINDEX_EMBED_BACKEND", raising=False)
    from reindex import config, embedding
    with pytest.raises(embedding.EmbeddingConfigError, match="cloudflare, ollama, openai"):
        config.resolve_embedding(None, None, None, tmp_path)


# ---------------------------------------------------------------------------
# resolve_duet: CLI flag > $CSINDEX_DUET > [reindex.sources].duet > True
# ---------------------------------------------------------------------------

@pytest.fixture
def _no_duet_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CSINDEX_DUET", raising=False)


def _write_duet_toml(root: Path, value: str) -> None:
    (root / "reindex.toml").write_text(f"[reindex.sources]\nduet = {value}\n", encoding="utf-8")


@pytest.mark.usefixtures("_no_duet_env")
def test_duet_defaults_on_without_toml(tmp_path: Path):
    assert config.resolve_duet(tmp_path) is True


@pytest.mark.usefixtures("_no_duet_env")
def test_duet_defaults_on_with_unrelated_toml(tmp_path: Path):
    (tmp_path / "reindex.toml").write_text('[reindex]\nprovider = "ollama"\n', encoding="utf-8")
    assert config.resolve_duet(tmp_path) is True


@pytest.mark.usefixtures("_no_duet_env")
@pytest.mark.parametrize(("toml_value", "expected"), [("false", False), ("true", True)])
def test_duet_toml_sets_value(tmp_path: Path, toml_value: str, expected: bool):
    _write_duet_toml(tmp_path, toml_value)
    assert config.resolve_duet(tmp_path) is expected


@pytest.mark.usefixtures("_no_duet_env")
def test_duet_toml_non_boolean_exits(tmp_path: Path):
    _write_duet_toml(tmp_path, '"off"')
    with pytest.raises(SystemExit, match=r"\[reindex\.sources\]"):
        config.resolve_duet(tmp_path)


@pytest.mark.parametrize("value", ["0", "false", "False", "NO", "off", " Off "])
def test_duet_env_disables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("CSINDEX_DUET", value)
    assert config.resolve_duet(tmp_path) is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_duet_env_enables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("CSINDEX_DUET", value)
    _write_duet_toml(tmp_path, "false")  # env beats toml
    assert config.resolve_duet(tmp_path) is True


@pytest.mark.parametrize("value", ["fasle", "disabled", "n", "2", "enable"])
def test_duet_env_unrecognized_value_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str):
    """A typo must not silently index (and bill) Duet: fail closed with the accepted values."""
    monkeypatch.setenv("CSINDEX_DUET", value)
    _write_duet_toml(tmp_path, "false")
    with pytest.raises(SystemExit) as exc_info:
        config.resolve_duet(tmp_path)
    msg = str(exc_info.value)
    assert "CSINDEX_DUET" in msg
    assert repr(value) in msg
    assert "1, true, yes, on" in msg
    assert "0, false, no, off" in msg


@pytest.mark.usefixtures("_no_duet_env")
def test_duet_unknown_sources_key_exits_naming_valid_keys(tmp_path: Path):
    (tmp_path / "reindex.toml").write_text("[reindex.sources]\ndeut = false\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        config.resolve_duet(tmp_path)
    msg = str(exc_info.value)
    assert "'deut'" in msg
    assert "[reindex.sources]" in msg
    assert "valid keys: duet" in msg


@pytest.mark.usefixtures("_no_duet_env")
@pytest.mark.parametrize("toml_text", ['[reindex]\nsources = "x"\n', "[reindex]\nsources = 1\n"])
def test_duet_sources_must_be_a_table(tmp_path: Path, toml_text: str):
    (tmp_path / "reindex.toml").write_text(toml_text, encoding="utf-8")
    with pytest.raises(SystemExit, match=r"\[reindex\.sources\] in reindex\.toml must be a table"):
        config.resolve_duet(tmp_path)


# A8: one info line when Duet runs on the default, so the default is never silent.

@pytest.fixture
def config_log(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    logger = MagicMock()
    monkeypatch.setattr(config.log, "get", lambda component: logger)
    return logger


def _events(logger: MagicMock) -> list[str]:
    return [c.args[0] for c in logger.info.call_args_list]


@pytest.mark.usefixtures("_no_duet_env")
def test_duet_default_logs_one_info_line_when_duet_exists(tmp_path: Path, config_log: MagicMock):
    (tmp_path / "duet").mkdir()
    assert config.resolve_duet(tmp_path) is True
    assert _events(config_log) == ["duet_default_on"]
    hint = config_log.info.call_args.kwargs["hint"]
    assert "LLM provider" in hint
    assert "embedding backend" in hint


@pytest.mark.usefixtures("_no_duet_env")
def test_duet_default_is_silent_without_a_duet_dir(tmp_path: Path, config_log: MagicMock):
    assert config.resolve_duet(tmp_path) is True
    assert _events(config_log) == []


@pytest.mark.parametrize("how", ["cli", "env", "toml"])
def test_duet_explicit_setting_never_logs_the_default_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_log: MagicMock, how: str,
):
    monkeypatch.delenv("CSINDEX_DUET", raising=False)
    (tmp_path / "duet").mkdir()
    cli_value = None
    if how == "cli":
        cli_value = True
    elif how == "env":
        monkeypatch.setenv("CSINDEX_DUET", "on")
    else:
        _write_duet_toml(tmp_path, "true")
    assert config.resolve_duet(tmp_path, cli_value) is True
    assert _events(config_log) == []


def test_duet_empty_env_is_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CSINDEX_DUET", "")
    _write_duet_toml(tmp_path, "false")
    assert config.resolve_duet(tmp_path) is False


def test_duet_env_beats_toml_when_disabling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CSINDEX_DUET", "off")
    _write_duet_toml(tmp_path, "true")
    assert config.resolve_duet(tmp_path) is False


def test_duet_cli_beats_env_and_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CSINDEX_DUET", "off")
    _write_duet_toml(tmp_path, "false")
    assert config.resolve_duet(tmp_path, True) is True
    monkeypatch.setenv("CSINDEX_DUET", "on")
    _write_duet_toml(tmp_path, "true")
    assert config.resolve_duet(tmp_path, False) is False
