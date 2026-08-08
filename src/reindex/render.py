"""
Render a Pydantic step model to INDEX.md (frontmatter + body).
Single source of truth for output layout. Adding a field = update model + this.

Cache-key fields (slug, type, content_hash/children_hash/inputs_hash,
generated_at, model) are emitted as STAMPED_AFTER and overwritten by stamp.py.
Leaves additionally emit conversation_model as STAMPED_AFTER -- the model the
original conversation ran on, read from the sibling README by stamp.py (not a
cache-key field).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

from reindex.models import LeafSummary, ProjectAggregate, RootAggregate


def _yaml_id_list(items: Sequence[str]) -> str:
    if not items:
        return "[]"
    return "[" + ", ".join(items) + "]"


def _yaml_string_list(items: Sequence[str]) -> str:
    if not items:
        return "[]"
    return "[" + ", ".join(json.dumps(s) for s in items) + "]"


def _bullet_list(items: Sequence[str], empty_text: str = "_(none)_") -> str:
    if not items:
        return empty_text
    return "\n".join(f"- {it}" for it in items)


# ---------------------------------------------------------------------------
# Leaf
# ---------------------------------------------------------------------------

def render_leaf(m: LeafSummary, out_file: Path) -> None:
    buf = StringIO()
    w = buf.write

    # Frontmatter (cache-key fields stamped post-render).
    w("---\n")
    w("slug: STAMPED_AFTER\n")
    w("type: conversation\n")
    w("content_hash: STAMPED_AFTER\n")
    w("generated_at: STAMPED_AFTER\n")
    w("model: STAMPED_AFTER\n")
    w("conversation_model: STAMPED_AFTER\n")
    w(f"title: {m.title}\n")
    w(f"conversation_type: {m.conversation_type}\n")
    w(f"outcome: {m.outcome}\n")
    w(f"complexity: {m.complexity}\n")
    w(f"reusability: {m.reusability}\n")
    w(f"natural_language: {m.natural_language}\n")
    w(f"has_code: {str(m.has_code).lower()}\n")
    w(f"turn_count: {m.turn_count}\n")
    w(f"date_range: {m.date_range_start.isoformat()} to {m.date_range_end.isoformat()}\n")
    w(f"topics: {_yaml_id_list(m.topics)}\n")
    w(f"code_languages: {_yaml_id_list(m.code_languages)}\n")
    w(f"tech_stack: {_yaml_id_list(m.tech_stack)}\n")
    w(f"privacy_flags: {_yaml_id_list(m.privacy_flags)}\n")
    w(f"entities: {_yaml_string_list(m.entities)}\n")
    w(f"artifacts: {_yaml_string_list(m.artifacts)}\n")
    w(f"semantic_keywords: {_yaml_string_list(m.semantic_keywords)}\n")
    w("---\n\n")

    # Body sections.
    w(f"## Summary\n{m.summary}\n\n")
    w(f"## Embedding\n{m.embedding_text}\n\n")
    w("## Key Points\n")
    w(_bullet_list(m.key_points) + "\n\n")
    w("## Outputs\n")
    w(_bullet_list(m.outputs) + "\n\n")
    w("## Decisions\n")
    w(_bullet_list(m.decisions) + "\n\n")
    w("## Action Items\n")
    w(_bullet_list(m.action_items) + "\n\n")
    w("## Unresolved Questions\n")
    w(_bullet_list(m.unresolved_questions) + "\n\n")
    w("## Concepts Introduced\n")
    if m.concepts_introduced:
        w("\n".join(f"- **{c.name}** — {c.brief}" for c in m.concepts_introduced) + "\n\n")
    else:
        w("_(none)_\n\n")
    w("## Citations\n")
    if m.citations:
        for c in m.citations:
            t = c.title or ""
            w(f"- [{c.type}] {t} — {c.ref}\n")
    else:
        w("_(none)_\n")

    out_file.write_text(buf.getvalue(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

def render_project(m: ProjectAggregate, out_file: Path) -> None:
    buf = StringIO()
    w = buf.write

    w("---\n")
    w("slug: STAMPED_AFTER\n")
    w("type: project\n")
    w("children_hash: STAMPED_AFTER\n")
    w("generated_at: STAMPED_AFTER\n")
    w("model: STAMPED_AFTER\n")
    w(f"project_status: {m.project_status}\n")
    w(f"velocity: {m.velocity}\n")
    w(f"dominant_outcome: {m.dominant_outcome}\n")
    w(f"conversation_count: {m.conversation_count}\n")
    w(f"knowledge_count: {m.knowledge_count}\n")
    w(f"date_range: {m.date_range_start.isoformat()} to {m.date_range_end.isoformat()}\n")
    w(f"topics: {_yaml_id_list(m.topics)}\n")
    w("tech_stack:\n")
    if not m.tech_stack:
        w("  []\n")
    else:
        for tc in m.tech_stack:
            w(f"  - {{ name: {tc.name}, count: {tc.count} }}\n")
    w("---\n\n")

    w(f"## Summary\n{m.summary}\n\n")
    w(f"## Embedding\n{m.embedding_text}\n\n")

    w("## Conversations\n")
    if m.conversations:
        for c in sorted(m.conversations, key=lambda x: x.slug):
            w(f"- {c.slug} — {c.title} — {c.gist}\n")
    else:
        w("_(none)_\n")
    w("\n## Knowledge Files\n")
    if m.knowledge_files:
        for kf in m.knowledge_files:
            w(f"- {kf.filename} — {kf.description}\n")
    else:
        w("_(none)_\n")
    w("\n## Recurring Themes\n")
    w(_bullet_list(m.recurring_themes) + "\n\n")
    w("## Open Action Items\n")
    if m.open_action_items:
        for ai in m.open_action_items:
            w(f"- ({ai.from_slug}) {ai.item}\n")
    else:
        w("_(none)_\n")

    out_file.write_text(buf.getvalue(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

def render_root(m: RootAggregate, out_file: Path) -> None:
    buf = StringIO()
    w = buf.write

    w("---\n")
    w("type: root\n")
    w("inputs_hash: STAMPED_AFTER\n")
    w("generated_at: STAMPED_AFTER\n")
    w("model: STAMPED_AFTER\n")
    w(f"project_count: {m.project_count}\n")
    w(f"conversation_count: {m.conversation_count}\n")
    w(f"date_range: {m.date_range_start.isoformat()} to {m.date_range_end.isoformat()}\n")
    w(f"top_topics: {_yaml_id_list(m.top_topics)}\n")
    w("---\n\n")

    w(f"## Overview\n{m.overview}\n\n")
    w(f"## Embedding\n{m.embedding_text}\n\n")

    w("## Projects\n")
    if m.projects:
        for p in sorted(m.projects, key=lambda x: x.slug):
            w(f"- {p.slug} — {p.gist}\n")
    else:
        w("_(none)_\n")
    w("\n## Top Themes\n")
    w(_bullet_list(m.top_themes) + "\n\n")
    w(f"## Standalone Conversations\n{m.standalone_overview}\n\n")

    w("## Time Distribution\n")
    if m.time_distribution:
        for tb in m.time_distribution:
            w(f"- {tb.year_month}: {tb.count}\n")
    else:
        w("_(empty)_\n")
    w("\n## Top Entities\n")
    if m.top_entities:
        for e in m.top_entities:
            w(f"- {e.name} ({e.count})\n")
    else:
        w("_(none)_\n")
    w("\n## Top Citations\n")
    if m.top_citations:
        for c in m.top_citations:
            t = c.title or ""
            w(f"- {t} — {c.ref} (×{c.count})\n")
    else:
        w("_(none)_\n")
    w("\n## Tech Stack Timeline\n")
    if m.tech_stack_timeline:
        for tt in sorted(m.tech_stack_timeline, key=lambda x: x.first_seen):
            w(f"- {tt.tech}: {tt.first_seen.isoformat()} → {tt.last_seen.isoformat()} (×{tt.count})\n")
    else:
        w("_(none)_\n")
    w("\n## Knowledge Clusters\n")
    if m.knowledge_clusters:
        for kc in m.knowledge_clusters:
            w(f"- **{kc.name}** (×{kc.conversation_count}) — sample topics: {', '.join(kc.sample_topics)}\n")
    else:
        w("_(none)_\n")

    out_file.write_text(buf.getvalue(), encoding="utf-8")
