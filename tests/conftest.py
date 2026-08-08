"""Shared pytest fixtures.

Strategy:
  * `tmp_export` — mints a self-contained export root in a tmp dir (with
    conversations/, projects/) so tests don't touch real data. Prompt
    templates are package data (see `reindex.prompt_loader`), not part
    of the export tree.
  * `valid_leaf_payload` / `valid_project_payload` / `valid_root_payload`
    — return validated Pydantic instances for render/finalize tests.
  * `mock_anthropic_client` — patches the shared AsyncAnthropic client in
    `reindex.providers.anthropic_api` with a configurable AsyncMock.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Filesystem fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Self-contained export root with empty conversations/projects dirs.

    Patches `reindex.paths.EXPORT_ROOT` (the single canonical attribute — all
    modules read it via `paths.EXPORT_ROOT` at call time) plus the cost/failure
    log env vars so tests are isolated. Prompt templates are package data
    (see `reindex.prompt_loader`), not part of the export tree, so they are
    not seeded here.
    """
    (tmp_path / "conversations").mkdir()
    (tmp_path / "projects").mkdir()

    from reindex import paths
    monkeypatch.setattr(paths, "EXPORT_ROOT", tmp_path)

    # Runtime root resolution (paths.require_export_root(), called by every
    # CLI command) recomputes EXPORT_ROOT from --root / $CSINDEX_ROOT / CWD,
    # which would otherwise clobber the direct EXPORT_ROOT patch above the
    # moment a command runs. Set both so it resolves back to this tree:
    #   - $CSINDEX_ROOT survives the CLI's `--root` callback, which always
    #     runs (even with no --root flag) and resets _requested_cli_root.
    #   - _requested_cli_root covers direct paths.resolve_root()/
    #     require_export_root() calls made outside the CLI app.
    monkeypatch.setenv("CSINDEX_ROOT", str(tmp_path))
    monkeypatch.setattr(paths, "_requested_cli_root", tmp_path)

    # Isolate cost + failure logs to the tmp tree.
    monkeypatch.setenv("CSINDEX_COST_LOG", str(tmp_path / ".reindex-costs.jsonl"))
    monkeypatch.setenv("CSINDEX_FAILURE_LOG", str(tmp_path / ".reindex-failures.jsonl"))

    # Test isolation: provider selection reads $CSINDEX_PROVIDER; clear it so
    # a developer's shell env can't leak into tests.
    monkeypatch.delenv("CSINDEX_PROVIDER", raising=False)

    # Init both logs empty so failures.count() / cost_log.aggregate() see 0.
    (tmp_path / ".reindex-costs.jsonl").write_text("", encoding="utf-8")
    (tmp_path / ".reindex-failures.jsonl").write_text("", encoding="utf-8")

    return tmp_path


@pytest.fixture
def make_conv(tmp_export: Path):
    """Factory for creating a conversation directory with conversation.md."""
    default_content = (
        "## Human\n_2024-08-23T14:23:22Z_\n\nhi\n\n"
        "## Assistant\n_2024-08-23T14:23:23Z_\n\nyo\n"
    )

    def _make(slug: str, content: str = default_content, *, project: str | None = None) -> Path:
        if project is not None:
            base = tmp_export / "projects" / project / "conversations" / slug
        else:
            base = tmp_export / "conversations" / slug
        base.mkdir(parents=True, exist_ok=True)
        (base / "conversation.md").write_text(content, encoding="utf-8")
        return base
    return _make


@pytest.fixture
def make_project(tmp_export: Path):
    def _make(slug: str, *, knowledge: dict[str, str] | None = None) -> Path:
        proj = tmp_export / "projects" / slug
        (proj / "conversations").mkdir(parents=True, exist_ok=True)
        if knowledge:
            (proj / "knowledge").mkdir(exist_ok=True)
            for fn, content in knowledge.items():
                (proj / "knowledge" / fn).write_text(content, encoding="utf-8")
        return proj
    return _make


# ---------------------------------------------------------------------------
# Pydantic payload fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_leaf_dict() -> dict:
    """Minimal-valid LeafSummary in dict form (matches schema requirements)."""
    return {
        "title": "test conversation",
        "summary": "User asked. Assistant answered. Issue resolved.",
        "embedding_text": "Test conversation about a thing. Includes the resolution.",
        "topics": ["topic-a", "topic-b", "topic-c"],
        "semantic_keywords": ["one", "two", "three", "four", "five"],
        "key_points": ["point 1"],
        "outputs": [],
        "turn_count": 2,
        "date_range_start": "2024-08-23",
        "date_range_end": "2024-08-23",
        "artifacts": [],
        "conversation_type": "how-to",
        "outcome": "resolved",
        "complexity": "simple",
        "reusability": "low",
        "tech_stack": [],
        "code_languages": [],
        "has_code": False,
        "entities": [],
        "citations": [],
        "concepts_introduced": [],
        "action_items": [],
        "unresolved_questions": [],
        "decisions": [],
        "privacy_flags": [],
        "natural_language": "en",
    }


@pytest.fixture
def valid_leaf(valid_leaf_dict):
    from reindex.models import LeafSummary
    return LeafSummary.model_validate(valid_leaf_dict)


@pytest.fixture
def valid_project_dict() -> dict:
    return {
        "summary": "Project does X.",
        "embedding_text": "Project description for embedding.",
        "conversations": [{"slug": "foo", "title": "foo title", "gist": "g"}],
        "knowledge_files": [],
        "recurring_themes": ["theme1"],
        "topics": ["topic-a"],
        "conversation_count": 1,
        "knowledge_count": 0,
        "date_range_start": "2024-01-01",
        "date_range_end": "2024-12-31",
        "project_status": "active",
        "velocity": "steady",
        "dominant_outcome": "resolved",
        "tech_stack": [],
        "open_action_items": [],
    }


@pytest.fixture
def valid_project(valid_project_dict):
    from reindex.models import ProjectAggregate
    return ProjectAggregate.model_validate(valid_project_dict)


@pytest.fixture
def valid_root_dict() -> dict:
    return {
        "overview": "Corpus contains things.",
        "embedding_text": "Top-level corpus overview.",
        "projects": [{"slug": "p1", "gist": "p1 gist"}],
        "top_themes": ["theme1"],
        "standalone_overview": "Some standalones.",
        "top_topics": ["topic-a"],
        "project_count": 1,
        "conversation_count": 1,
        "date_range_start": "2024-01-01",
        "date_range_end": "2024-12-31",
        "time_distribution": [{"year_month": "2024-08", "count": 1}],
        "top_entities": [{"name": "Foo", "count": 3}],
        "top_citations": [],
        "tech_stack_timeline": [],
        "knowledge_clusters": [],
    }


@pytest.fixture
def valid_root(valid_root_dict):
    from reindex.models import RootAggregate
    return RootAggregate.model_validate(valid_root_dict)


# ---------------------------------------------------------------------------
# Anthropic SDK mocks
# ---------------------------------------------------------------------------

class FakeToolUseBlock:
    """Mimics anthropic.types.ToolUseBlock for type checks via isinstance."""
    def __init__(self, input_dict: dict, id: str = "toolu_test"):
        self.type = "tool_use"
        self.id = id
        self.input = input_dict


class FakeUsage:
    def __init__(self, input_tokens: int = 100, output_tokens: int = 50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeMessage:
    def __init__(self, content: list, usage: FakeUsage | None = None):
        self.content = content
        self.usage = usage or FakeUsage()


def make_messages_create_response(payload_dict: dict) -> FakeMessage:
    """Build a FakeMessage that mimics a successful tool_use response."""
    return FakeMessage(content=[FakeToolUseBlock(input_dict=payload_dict)])


@pytest.fixture
def mock_anthropic_client(monkeypatch: pytest.MonkeyPatch):
    """Patches anthropic.AsyncAnthropic with a configurable AsyncMock.

    Returns a MagicMock with attributes:
      * messages.create — AsyncMock
      * messages.batches.create — AsyncMock
      * messages.batches.retrieve — AsyncMock
      * messages.batches.results — AsyncMock
      * close — AsyncMock
    """
    from reindex.providers import anthropic_api

    client = MagicMock()
    client.messages.create = AsyncMock()
    client.messages.batches.create = AsyncMock()
    client.messages.batches.retrieve = AsyncMock()
    client.messages.batches.results = AsyncMock()
    client.close = AsyncMock()

    # Client ownership lives in providers.anthropic_api; backend.get_async_client
    # delegates there at call time, so this one patch covers both names.
    monkeypatch.setattr(anthropic_api, "_async_client", client)
    monkeypatch.setattr(anthropic_api, "get_async_client", lambda: client)

    yield client


@pytest.fixture
def fake_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake")


# ---------------------------------------------------------------------------
# Shutdown module isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_shutdown_state():
    """Reset the module-level shutdown._controller singleton after every test.

    Tests that exercise code paths that call shutdown.install() (e.g. batches
    resume) leave _controller set.  Without this reset the next test that
    asserts ``shutdown.get() is None`` fails spuriously due to ordering.
    """
    from reindex import shutdown
    yield
    shutdown._controller = None
