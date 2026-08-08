"""coerce: pre-validation shape fixes for unguided LLM tool outputs."""

from __future__ import annotations

from reindex.coerce import coerce_for_model
from reindex.models import LeafSummary, ProjectAggregate

# ---------------------------------------------------------------------------
# String-where-array
# ---------------------------------------------------------------------------

def test_string_wraps_to_list(valid_leaf_dict):
    """Common LLM bug: returning a string for a list[str] field."""
    payload = dict(valid_leaf_dict)
    payload["entities"] = "Hashicorp"   # should be list
    fixed = coerce_for_model(payload, LeafSummary)
    assert fixed["entities"] == ["Hashicorp"]
    LeafSummary.model_validate(fixed)


def test_empty_string_becomes_empty_list(valid_leaf_dict):
    payload = dict(valid_leaf_dict)
    payload["action_items"] = ""
    fixed = coerce_for_model(payload, LeafSummary)
    assert fixed["action_items"] == []
    LeafSummary.model_validate(fixed)


def test_null_becomes_empty_list(valid_leaf_dict):
    payload = dict(valid_leaf_dict)
    payload["citations"] = None
    fixed = coerce_for_model(payload, LeafSummary)
    assert fixed["citations"] == []
    LeafSummary.model_validate(fixed)


def test_extra_keys_dropped(valid_leaf_dict):
    payload = dict(valid_leaf_dict)
    payload["unknown_field"] = "garbage"
    fixed = coerce_for_model(payload, LeafSummary)
    assert "unknown_field" not in fixed
    LeafSummary.model_validate(fixed)


def test_nested_object_string_wrap(valid_project_dict):
    """tech_stack is list[TechCount]; if model returns one as a dict not in list,
    we wrap. (Not common, but defensive.)"""
    payload = dict(valid_project_dict)
    payload["tech_stack"] = {"name": "x", "count": 1}  # should be list of objects
    fixed = coerce_for_model(payload, ProjectAggregate)
    assert isinstance(fixed["tech_stack"], list)
    assert len(fixed["tech_stack"]) == 1


# ---------------------------------------------------------------------------
# Date sentinel coercion (handled in models, not coerce — but pipeline still works)
# ---------------------------------------------------------------------------

def test_date_unknown_sentinel(valid_leaf_dict):
    valid_leaf_dict["date_range_start"] = "<UNKNOWN>"
    valid_leaf_dict["date_range_end"] = "n/a"
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.date_range_start.year == 1970
    assert m.date_range_end.year == 1970


def test_date_real_passes_through(valid_leaf_dict):
    valid_leaf_dict["date_range_start"] = "2024-08-23"
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.date_range_start.isoformat() == "2024-08-23"


# ---------------------------------------------------------------------------
# Optional arrays default to []
# ---------------------------------------------------------------------------

def test_omitted_optional_arrays_default_empty(valid_leaf_dict):
    """Drop a bunch of optional arrays — should still validate with [] defaults."""
    for key in ("outputs", "artifacts", "tech_stack", "code_languages",
                "entities", "citations", "concepts_introduced",
                "action_items", "unresolved_questions", "decisions",
                "privacy_flags"):
        valid_leaf_dict.pop(key, None)
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.outputs == []
    assert m.citations == []
    assert m.privacy_flags == []


# ---------------------------------------------------------------------------
# Soft cap removed
# ---------------------------------------------------------------------------

def test_topics_over_8_now_allowed(valid_leaf_dict):
    valid_leaf_dict["topics"] = [f"t-{i}" for i in range(20)]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert len(m.topics) == 20


def test_semantic_keywords_over_30_now_allowed(valid_leaf_dict):
    valid_leaf_dict["semantic_keywords"] = [f"kw{i}" for i in range(50)]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert len(m.semantic_keywords) == 50


def test_topics_min_floor_relaxed(valid_leaf_dict):
    """Floor was demoted to a soft signal; a single-item topics list now validates."""
    valid_leaf_dict["topics"] = ["only-one"]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.topics == ["only-one"]


# ---------------------------------------------------------------------------
# CSV-split heuristic: tag-like CSV inside a 1-element list gets split
# ---------------------------------------------------------------------------

def test_csv_split_topics_real_prod_pattern(valid_leaf_dict):
    """Prod failure: model returns ["a, b, c, d"] for topics; coerce splits."""
    payload = dict(valid_leaf_dict)
    payload["topics"] = [
        "particle-physics, quantum-field-theory, one-electron-universe, quantum-mechanics"
    ]
    fixed = coerce_for_model(payload, LeafSummary)
    assert fixed["topics"] == [
        "particle-physics",
        "quantum-field-theory",
        "one-electron-universe",
        "quantum-mechanics",
    ]
    LeafSummary.model_validate(fixed)


def test_csv_split_does_not_split_sentence(valid_leaf_dict):
    """Free-text-looking single string should NOT be CSV-split."""
    payload = dict(valid_leaf_dict)
    payload["key_points"] = ["hello world. This is a sentence."]
    fixed = coerce_for_model(payload, LeafSummary)
    assert fixed["key_points"] == ["hello world. This is a sentence."]


def test_csv_split_skipped_in_unsafe_field(valid_leaf_dict):
    """`decisions` is NOT in _CSV_SAFE_FIELDS — CSV-looking string stays intact."""
    payload = dict(valid_leaf_dict)
    payload["decisions"] = ["a, b, c"]
    fixed = coerce_for_model(payload, LeafSummary)
    assert fixed["decisions"] == ["a, b, c"]


def test_csv_split_then_validate_end_to_end(valid_leaf_dict):
    """CSV string in topics gets split and downstream validation succeeds."""
    payload = dict(valid_leaf_dict)
    payload["topics"] = "topic-a, topic-b, topic-c"  # string-where-array, CSV
    fixed = coerce_for_model(payload, LeafSummary)
    assert fixed["topics"] == ["topic-a", "topic-b", "topic-c"]
    m = LeafSummary.model_validate(fixed)
    assert m.topics == ["topic-a", "topic-b", "topic-c"]


def test_coerce_preserves_valid_payload(valid_leaf_dict):
    """A perfectly valid payload should pass through coerce unchanged."""
    fixed = coerce_for_model(valid_leaf_dict, LeafSummary)
    LeafSummary.model_validate(fixed)
    assert fixed["topics"] == valid_leaf_dict["topics"]


# ---------------------------------------------------------------------------
# Expanded enums
# ---------------------------------------------------------------------------

def test_other_conversation_type_now_valid(valid_leaf_dict):
    valid_leaf_dict["conversation_type"] = "other"
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.conversation_type == "other"


def test_other_outcome_now_valid(valid_leaf_dict):
    valid_leaf_dict["outcome"] = "other"
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.outcome == "other"
