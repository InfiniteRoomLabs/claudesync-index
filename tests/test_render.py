"""Render: Pydantic -> INDEX.md frontmatter + body."""

from __future__ import annotations

from pathlib import Path

from reindex.render import render_leaf, render_project, render_root

# ---------------------------------------------------------------------------
# Leaf
# ---------------------------------------------------------------------------

def test_render_leaf_writes_frontmatter_and_body(tmp_path: Path, valid_leaf):
    out = tmp_path / "INDEX.md"
    render_leaf(valid_leaf, out)
    content = out.read_text(encoding="utf-8")
    # STAMPED_AFTER for cache-key fields (gets overwritten later by stamp.py).
    assert "content_hash: STAMPED_AFTER" in content
    assert "slug: STAMPED_AFTER" in content
    assert "type: conversation" in content
    assert "## Summary" in content
    assert valid_leaf.summary in content
    assert "## Embedding" in content
    assert valid_leaf.embedding_text in content


def test_render_leaf_topics_inline(tmp_path: Path, valid_leaf):
    out = tmp_path / "INDEX.md"
    render_leaf(valid_leaf, out)
    content = out.read_text(encoding="utf-8")
    assert "topics: [topic-a, topic-b, topic-c]" in content


def test_render_leaf_empty_arrays_emit_brackets(tmp_path: Path, valid_leaf):
    out = tmp_path / "INDEX.md"
    render_leaf(valid_leaf, out)
    content = out.read_text(encoding="utf-8")
    assert "code_languages: []" in content
    assert "tech_stack: []" in content
    assert "artifacts: []" in content


def test_render_leaf_empty_outputs_shows_none(tmp_path: Path, valid_leaf):
    out = tmp_path / "INDEX.md"
    render_leaf(valid_leaf, out)
    content = out.read_text(encoding="utf-8")
    assert "## Outputs" in content
    assert "_(none)_" in content


def test_render_leaf_with_concepts_and_citations(tmp_path: Path, valid_leaf_dict):
    from reindex.models import LeafSummary
    valid_leaf_dict["concepts_introduced"] = [{"name": "Foo", "brief": "what it is"}]
    valid_leaf_dict["citations"] = [
        {"type": "url", "ref": "https://x.test", "title": "X"},
        {"type": "paper", "ref": "10.1234/abc", "title": None},
    ]
    m = LeafSummary.model_validate(valid_leaf_dict)
    out = tmp_path / "INDEX.md"
    render_leaf(m, out)
    content = out.read_text(encoding="utf-8")
    assert "**Foo** — what it is" in content
    assert "[url] X — https://x.test" in content
    assert "[paper]" in content
    assert "10.1234/abc" in content


def test_render_leaf_string_with_quotes(tmp_path: Path, valid_leaf_dict):
    """Entities that contain spaces/punctuation get JSON-quoted in YAML lists."""
    from reindex.models import LeafSummary
    valid_leaf_dict["entities"] = ["Hashi Corp", "Anthropic"]
    m = LeafSummary.model_validate(valid_leaf_dict)
    out = tmp_path / "INDEX.md"
    render_leaf(m, out)
    content = out.read_text(encoding="utf-8")
    assert '"Hashi Corp"' in content
    assert '"Anthropic"' in content


def test_render_leaf_has_code_lowercases_bool(tmp_path: Path, valid_leaf):
    out = tmp_path / "INDEX.md"
    render_leaf(valid_leaf, out)
    assert "has_code: false" in out.read_text(encoding="utf-8")


def test_render_leaf_date_range_iso(tmp_path: Path, valid_leaf):
    out = tmp_path / "INDEX.md"
    render_leaf(valid_leaf, out)
    assert "date_range: 2024-08-23 to 2024-08-23" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

def test_render_project_basic(tmp_path: Path, valid_project):
    out = tmp_path / "INDEX.md"
    render_project(valid_project, out)
    content = out.read_text(encoding="utf-8")
    assert "type: project" in content
    assert "children_hash: STAMPED_AFTER" in content
    assert valid_project.summary in content
    assert "## Conversations" in content
    assert "- foo — foo title — g" in content


def test_render_project_empty_arrays(tmp_path: Path, valid_project_dict):
    from reindex.models import ProjectAggregate
    valid_project_dict["conversations"] = []
    valid_project_dict["knowledge_files"] = []
    valid_project_dict["tech_stack"] = []
    valid_project_dict["recurring_themes"] = []
    valid_project_dict["open_action_items"] = []
    valid_project_dict["conversation_count"] = 0
    m = ProjectAggregate.model_validate(valid_project_dict)
    out = tmp_path / "INDEX.md"
    render_project(m, out)
    content = out.read_text(encoding="utf-8")
    assert "tech_stack:\n  []" in content
    assert content.count("_(none)_") >= 3  # conversations, knowledge_files, recurring_themes, open_action_items


def test_render_project_tech_stack_with_counts(tmp_path: Path, valid_project_dict):
    from reindex.models import ProjectAggregate
    valid_project_dict["tech_stack"] = [
        {"name": "symfony", "count": 5},
        {"name": "postgres", "count": 3},
    ]
    m = ProjectAggregate.model_validate(valid_project_dict)
    out = tmp_path / "INDEX.md"
    render_project(m, out)
    content = out.read_text(encoding="utf-8")
    assert "{ name: symfony, count: 5 }" in content
    assert "{ name: postgres, count: 3 }" in content


def test_render_project_open_action_items_with_backref(tmp_path: Path, valid_project_dict):
    from reindex.models import ProjectAggregate
    valid_project_dict["open_action_items"] = [
        {"from_slug": "convo-x", "item": "do thing"},
    ]
    m = ProjectAggregate.model_validate(valid_project_dict)
    out = tmp_path / "INDEX.md"
    render_project(m, out)
    assert "- (convo-x) do thing" in out.read_text(encoding="utf-8")


def test_render_project_conversations_alphabetical(tmp_path: Path, valid_project_dict):
    from reindex.models import ProjectAggregate
    valid_project_dict["conversations"] = [
        {"slug": "zebra", "title": "Z", "gist": "z"},
        {"slug": "alpha", "title": "A", "gist": "a"},
        {"slug": "mike", "title": "M", "gist": "m"},
    ]
    m = ProjectAggregate.model_validate(valid_project_dict)
    out = tmp_path / "INDEX.md"
    render_project(m, out)
    content = out.read_text(encoding="utf-8")
    a_idx = content.index("alpha")
    m_idx = content.index("mike")
    z_idx = content.index("zebra")
    assert a_idx < m_idx < z_idx


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

def test_render_root_basic(tmp_path: Path, valid_root):
    out = tmp_path / "INDEX.md"
    render_root(valid_root, out)
    content = out.read_text(encoding="utf-8")
    assert "type: root" in content
    assert "inputs_hash: STAMPED_AFTER" in content
    assert valid_root.overview in content
    assert "## Time Distribution" in content
    assert "- 2024-08: 1" in content


def test_render_root_top_entities(tmp_path: Path, valid_root):
    out = tmp_path / "INDEX.md"
    render_root(valid_root, out)
    assert "- Foo (3)" in out.read_text(encoding="utf-8")


def test_render_root_empty_clusters(tmp_path: Path, valid_root_dict):
    from reindex.models import RootAggregate
    valid_root_dict["knowledge_clusters"] = []
    valid_root_dict["tech_stack_timeline"] = []
    valid_root_dict["top_citations"] = []
    valid_root_dict["time_distribution"] = []
    m = RootAggregate.model_validate(valid_root_dict)
    out = tmp_path / "INDEX.md"
    render_root(m, out)
    content = out.read_text(encoding="utf-8")
    assert "_(empty)_" in content  # time distribution
    assert "_(none)_" in content


def test_render_root_tech_timeline_chronological(tmp_path: Path, valid_root_dict):
    from reindex.models import RootAggregate
    valid_root_dict["tech_stack_timeline"] = [
        {"tech": "redis", "first_seen": "2024-06-01", "last_seen": "2024-12-31", "count": 5},
        {"tech": "kubernetes", "first_seen": "2023-01-01", "last_seen": "2024-12-01", "count": 20},
        {"tech": "rust", "first_seen": "2024-09-01", "last_seen": "2024-12-31", "count": 3},
    ]
    m = RootAggregate.model_validate(valid_root_dict)
    out = tmp_path / "INDEX.md"
    render_root(m, out)
    content = out.read_text(encoding="utf-8")
    k = content.index("kubernetes")
    r1 = content.index("redis")
    r2 = content.index("rust")
    assert k < r1 < r2


def test_render_root_clusters(tmp_path: Path, valid_root_dict):
    from reindex.models import RootAggregate
    valid_root_dict["knowledge_clusters"] = [
        {"name": "Backend", "sample_topics": ["postgres", "redis"], "conversation_count": 12},
    ]
    m = RootAggregate.model_validate(valid_root_dict)
    out = tmp_path / "INDEX.md"
    render_root(m, out)
    content = out.read_text(encoding="utf-8")
    assert "**Backend** (×12)" in content
    assert "postgres, redis" in content
