"""
Provider registry. Plain dict — no entry points, no plugins.

Adding a provider: copy _template.py, implement invoke(), add one line
here, add a defaults block in config._BUILTIN_DEFAULTS, add a contract
test fixture in tests/test_provider_contract.py.
"""

from __future__ import annotations

from reindex.config import ProviderConfig, ProviderName, coerce_provider_name
from reindex.providers.base import BatchCapable, Provider

_REGISTRY: dict[ProviderName, type[Provider]] = {}


def _registry() -> dict[ProviderName, type[Provider]]:
    """Lazy-populated so importing reindex.providers doesn't drag every
    provider's SDK in (anthropic, httpx) for callers that need only one."""
    if not _REGISTRY:
        from reindex.providers.anthropic_api import AnthropicApiProvider
        from reindex.providers.claude_cli import ClaudeCliProvider
        from reindex.providers.ollama import OllamaProvider
        from reindex.providers.opencode import OpencodeProvider

        _REGISTRY[ProviderName.CLAUDE_CLI] = ClaudeCliProvider
        _REGISTRY[ProviderName.ANTHROPIC] = AnthropicApiProvider
        _REGISTRY[ProviderName.OLLAMA] = OllamaProvider
        _REGISTRY[ProviderName.OPENCODE] = OpencodeProvider
    return _REGISTRY


def provider_class(name: str | ProviderName) -> type[Provider]:
    name = coerce_provider_name(name)
    reg = _registry()
    try:
        return reg[name]
    except KeyError as e:
        raise SystemExit(
            f"provider {name.value!r} not implemented yet; available: {sorted(p.value for p in reg)}"
        ) from e


def get_provider(name: str | ProviderName, config: ProviderConfig) -> Provider:
    return provider_class(name)(config)


__all__ = ["BatchCapable", "Provider", "ProviderName", "get_provider", "provider_class"]
