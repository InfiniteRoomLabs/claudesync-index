"""Workers: prepare/finalize/serialize/restore for leaf, project, root."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reindex.workers import (
    LeafItem,
    ProjectItem,
    RootItem,
    TranscriptMode,
    finalize_leaf,
    finalize_persisted,
    finalize_project,
    finalize_root,
    prepare_leaf,
    prepare_project,
    prepare_root,
    serialize_leaf_item,
    serialize_project_item,
    serialize_root_item,
)

# ---------------------------------------------------------------------------
# prepare_leaf — cache hit / miss / skip / errors
# ---------------------------------------------------------------------------

def test_prepare_leaf_cache_miss(tmp_export, make_conv):
    conv = make_conv("alpha", "## Human\n_2024-01-01_\n\nhello\n")
    item = prepare_leaf(conv)
    assert item is not None
    assert item.slug == "alpha"
    assert item.content_hash != ""
    assert item.user_content.startswith("## Human")


def test_prepare_leaf_skips_empty_file(tmp_export, make_conv):
    conv = make_conv("empty", "")
    item = prepare_leaf(conv)
    assert item is None


def test_prepare_leaf_cache_hit(tmp_export, make_conv):
    conv = make_conv("foo", "content")
    first = prepare_leaf(conv)
    assert first is not None
    # Write an INDEX.md with the matching content_hash.
    (conv / "INDEX.md").write_text(
        f"---\nslug: foo\ncontent_hash: {first.content_hash}\n---\nbody\n",
        encoding="utf-8",
    )
    second = prepare_leaf(conv)
    assert second is None


def test_prepare_leaf_force_bypasses_cache(tmp_export, make_conv):
    conv = make_conv("foo", "content")
    first = prepare_leaf(conv)
    assert first is not None
    (conv / "INDEX.md").write_text(
        f"---\nslug: foo\ncontent_hash: {first.content_hash}\n---\nbody\n",
        encoding="utf-8",
    )
    forced = prepare_leaf(conv, force=True)
    assert forced is not None


def test_prepare_leaf_no_conversation_md(tmp_export):
    d = tmp_export / "conversations" / "broken"
    d.mkdir(parents=True)
    item = prepare_leaf(d)
    assert item is None


def test_prepare_leaf_picks_sonnet_for_huge(tmp_export, make_conv):
    conv = make_conv("big", "x" * 700_000)  # > 600KB threshold
    item = prepare_leaf(conv)
    assert item is not None
    assert "sonnet" in item.model


def test_prepare_leaf_picks_haiku_for_small(tmp_export, make_conv):
    conv = make_conv("tiny", "small")
    item = prepare_leaf(conv)
    assert item is not None
    assert "haiku" in item.model


# ---------------------------------------------------------------------------
# Transcript mode -- inline (default) vs file (huge transcript via Read tool)
# ---------------------------------------------------------------------------

def test_prepare_leaf_uses_inline_mode_under_threshold(tmp_export, make_conv):
    conv = make_conv("small", "x" * 1024)
    item = prepare_leaf(conv)
    assert item is not None
    assert item.transcript_mode is TranscriptMode.INLINE
    # user_content carries the full transcript inline.
    assert item.user_content.count("x") == 1024


def test_prepare_leaf_uses_file_mode_above_threshold(tmp_export, make_conv):
    # 210KB > 200KB inline threshold but < 600KB Sonnet threshold, so we
    # get the file-mode prompt while still running on Haiku.
    conv = make_conv("biggish", "x" * (210 * 1024))
    item = prepare_leaf(conv)
    assert item is not None
    assert item.transcript_mode is TranscriptMode.FILE
    # user_content should be a pointer, NOT the transcript.
    assert "x" * 1024 not in item.user_content
    assert "Read tool" in item.user_content
    assert str(conv / "conversation.md") in item.user_content


def test_prepare_leaf_inline_and_file_use_different_prompts(tmp_export, make_conv):
    small = prepare_leaf(make_conv("s", "tiny"))
    big = prepare_leaf(make_conv("b", "x" * (210 * 1024)))
    assert small is not None and big is not None
    assert small.system_prompt != big.system_prompt


def test_prepare_leaf_inlines_when_provider_cannot_read_files(tmp_export, make_conv):
    """Regression: a >200KB transcript on a provider without a Read tool
    must stay INLINE. Pre-provider code picked FILE by size alone, so the
    batch path shipped a prompt pointing at a file the API couldn't read."""
    from reindex import config
    from reindex.providers.base import Provider

    class NoFsProvider(Provider):
        name = "no-fs"
        supports_file_transcripts = False

        async def invoke(self, req):  # pragma: no cover - never called here
            raise NotImplementedError

    provider = NoFsProvider(config.load(tmp_export, provider_name="claude-cli"))
    conv = make_conv("big-api", "x" * (210 * 1024))
    item = prepare_leaf(conv, provider=provider)
    assert item is not None
    assert item.transcript_mode is TranscriptMode.INLINE
    assert "Read tool" not in item.user_content
    assert item.user_content.count("x") == 210 * 1024


# ---------------------------------------------------------------------------
# prepare_project — cache + content gathering
# ---------------------------------------------------------------------------

def test_prepare_project_cache_miss(tmp_export, make_project):
    p = make_project("proj1", knowledge={"spec.md": "design"})
    item = prepare_project(p)
    assert item is not None
    assert item.slug == "proj1"
    assert "knowledge/spec.md" in item.user_content


def test_prepare_project_cache_hit(tmp_export, make_project):
    p = make_project("proj1")
    item = prepare_project(p)
    assert item is not None
    (p / "INDEX.md").write_text(
        f"---\nslug: proj1\nchildren_hash: {item.children_hash}\n---\nbody\n",
        encoding="utf-8",
    )
    assert prepare_project(p) is None


def test_prepare_project_includes_child_index(tmp_export, make_project, make_conv):
    p = make_project("proj1")
    make_conv("c1", "content", project="proj1")
    (p / "conversations" / "c1" / "INDEX.md").write_text(
        "---\nslug: c1\ncontent_hash: abc\ntopics: [a, b, c]\n---\n\nbody\n",
        encoding="utf-8",
    )
    item = prepare_project(p)
    assert item is not None
    assert "=== c1 ===" in item.user_content
    assert "content_hash: abc" in item.user_content


# ---------------------------------------------------------------------------
# prepare_root
# ---------------------------------------------------------------------------

def test_prepare_root_cache_miss(tmp_export):
    item = prepare_root(tmp_export)
    assert item is not None
    assert item.inputs_hash != ""


def test_prepare_root_cache_hit(tmp_export):
    item = prepare_root(tmp_export)
    assert item is not None
    (tmp_export / "INDEX.md").write_text(
        f"---\ntype: root\ninputs_hash: {item.inputs_hash}\n---\nbody\n",
        encoding="utf-8",
    )
    assert prepare_root(tmp_export) is None


def test_prepare_root_force_bypasses_cache(tmp_export):
    item = prepare_root(tmp_export)
    assert item is not None
    (tmp_export / "INDEX.md").write_text(
        f"---\ntype: root\ninputs_hash: {item.inputs_hash}\n---\nbody\n",
        encoding="utf-8",
    )
    assert prepare_root(tmp_export, force=True) is not None


# ---------------------------------------------------------------------------
# finalize_X — render + stamp + cost log
# ---------------------------------------------------------------------------

def test_finalize_leaf_writes_index_and_stamps(tmp_export, make_conv, valid_leaf):
    conv = make_conv("foo", "content")
    # claudesync README carries the conversation's own model.
    (conv / "README.md").write_text(
        "# Foo\n\n- **Model:** claude-opus-4-7\n", encoding="utf-8"
    )
    item = LeafItem(
        slug="foo", conv_dir=conv, out_file=conv / "INDEX.md",
        content_hash="REAL_HASH" + "0" * 56, model="claude-haiku-4-5",
        generated_at="2026-01-01T00:00:00Z",
        system_prompt="", user_content="",
    )
    finalize_leaf(item, valid_leaf, cost=0.05, turns=1, duration_ms=1234)
    content = (conv / "INDEX.md").read_text(encoding="utf-8")
    assert "STAMPED_AFTER" not in content  # all stamped
    assert f"content_hash: {item.content_hash}" in content
    assert "slug: foo" in content
    # Summarizer model vs the conversation's own model are stamped distinctly.
    assert "model: claude-haiku-4-5" in content
    assert "conversation_model: claude-opus-4-7" in content


def test_finalize_leaf_conversation_model_unknown_without_readme(tmp_export, make_conv, valid_leaf):
    conv = make_conv("bar", "content")  # no README.md
    item = LeafItem(
        slug="bar", conv_dir=conv, out_file=conv / "INDEX.md",
        content_hash="H" * 64, model="claude-haiku-4-5",
        generated_at="2026-01-01T00:00:00Z",
        system_prompt="", user_content="",
    )
    finalize_leaf(item, valid_leaf, cost=0.0, turns=1, duration_ms=1)
    content = (conv / "INDEX.md").read_text(encoding="utf-8")
    assert "conversation_model: unknown" in content


def test_finalize_project_writes_index(tmp_export, make_project, valid_project):
    p = make_project("p1")
    item = ProjectItem(
        slug="p1", project_dir=p, out_file=p / "INDEX.md",
        children_hash="C" * 64, model="claude-haiku-4-5",
        generated_at="2026-01-01T00:00:00Z",
        system_prompt="", user_content="",
    )
    finalize_project(item, valid_project, cost=0.01, turns=1, duration_ms=100)
    content = (p / "INDEX.md").read_text(encoding="utf-8")
    assert f"children_hash: {'C' * 64}" in content
    assert "slug: p1" in content


def test_finalize_root_writes_index(tmp_export, valid_root):
    item = RootItem(
        out_file=tmp_export / "INDEX.md",
        inputs_hash="I" * 64, model="claude-opus-4-7",
        generated_at="2026-01-01T00:00:00Z",
        system_prompt="", user_content="",
    )
    finalize_root(item, valid_root, cost=1.0, turns=1, duration_ms=5000)
    content = (tmp_export / "INDEX.md").read_text(encoding="utf-8")
    assert f"inputs_hash: {'I' * 64}" in content
    assert "model: claude-opus-4-7" in content


def test_finalize_leaf_records_cost(tmp_export, make_conv, valid_leaf):
    conv = make_conv("foo", "content")
    item = LeafItem(
        slug="foo", conv_dir=conv, out_file=conv / "INDEX.md",
        content_hash="x" * 64, model="m",
        generated_at="t", system_prompt="", user_content="",
    )
    finalize_leaf(item, valid_leaf, cost=0.123, turns=2, duration_ms=999)

    log_path = Path(__import__("os").environ["CSINDEX_COST_LOG"])
    rec = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert rec["slug"] == "foo"
    assert rec["cost"] == 0.123


# ---------------------------------------------------------------------------
# serialize/restore round trip
# ---------------------------------------------------------------------------

def test_serialize_leaf_restore(tmp_export):
    item = LeafItem(
        slug="x", conv_dir=tmp_export / "conversations" / "x",
        out_file=tmp_export / "conversations" / "x" / "INDEX.md",
        content_hash="abc", model="m", generated_at="t",
        system_prompt="big", user_content="huge",
    )
    d = serialize_leaf_item(item)
    # JSON round-trip safe.
    json.dumps(d)
    # Prompts NOT persisted (they're not needed for finalize).
    assert "system_prompt" not in d
    assert "user_content" not in d
    assert d["content_hash"] == "abc"
    assert d["slug"] == "x"


def test_serialize_project_restore():
    item = ProjectItem(
        slug="p", project_dir=Path("/x/p"),
        out_file=Path("/x/p/INDEX.md"),
        children_hash="cc", model="m", generated_at="t",
        system_prompt="", user_content="",
    )
    d = serialize_project_item(item)
    json.dumps(d)
    assert d["children_hash"] == "cc"


def test_serialize_root_restore():
    item = RootItem(
        out_file=Path("/INDEX.md"),
        inputs_hash="ii", model="opus", generated_at="t",
        system_prompt="", user_content="",
    )
    d = serialize_root_item(item)
    json.dumps(d)
    assert d["inputs_hash"] == "ii"


# ---------------------------------------------------------------------------
# finalize_persisted (resume path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_finalize_persisted_leaf(tmp_export, make_conv, valid_leaf):
    conv = make_conv("foo", "content")
    kwargs = {
        "out_file": str(conv / "INDEX.md"),
        "content_hash": "z" * 64,
        "slug": "foo",
        "generated_at": "2026-01-01T00:00:00Z",
        "model": "claude-haiku-4-5",
    }
    await finalize_persisted("leaf", kwargs, valid_leaf, cost=0.01, turns=1, duration_ms=10)
    assert (conv / "INDEX.md").is_file()
    assert f"content_hash: {'z' * 64}" in (conv / "INDEX.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_finalize_persisted_unknown_step_raises(valid_leaf):
    with pytest.raises(ValueError, match="unknown step"):
        await finalize_persisted("nope", {}, valid_leaf, 0, 0, 0)
