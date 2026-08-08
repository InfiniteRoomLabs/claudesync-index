"""Pydantic model validation: happy paths, edge cases, negative paths."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from reindex.models import (
    STEP_MODEL,
    LeafSummary,
    ProjectAggregate,
    RootAggregate,
    schema_for,
)

# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_leaf_validates_minimal(valid_leaf_dict):
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.title == "test conversation"
    assert m.outcome == "resolved"


def test_project_validates_minimal(valid_project_dict):
    m = ProjectAggregate.model_validate(valid_project_dict)
    assert m.conversation_count == 1


def test_root_validates_minimal(valid_root_dict):
    m = RootAggregate.model_validate(valid_root_dict)
    assert m.project_count == 1


def test_step_model_lookup():
    assert STEP_MODEL["leaf"] is LeafSummary
    assert STEP_MODEL["project"] is ProjectAggregate
    assert STEP_MODEL["root"] is RootAggregate


def test_schema_for_returns_json_schema():
    s = schema_for("leaf")
    assert s["type"] == "object"
    assert "title" in s["properties"]
    assert s["properties"]["title"]["type"] == "string"


# ---------------------------------------------------------------------------
# Enum fallback: unknown conversation_type / outcome -> 'other' instead of raise
# ---------------------------------------------------------------------------

def test_outcome_unknown_falls_back_to_other(valid_leaf_dict):
    # Used to raise; the mode="before" enum-fallback validator now coerces
    # to the explicit 'other' escape hatch instead of failing the whole
    # summary over a single bad enum.
    valid_leaf_dict["outcome"] = "magical"
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.outcome == "other"


def test_conversation_type_unknown_falls_back_to_other(valid_leaf_dict):
    valid_leaf_dict["conversation_type"] = "unknown-type"
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.conversation_type == "other"


def test_conversation_type_crosswire_falls_back(valid_leaf_dict):
    # Observed in production: model emitted an Outcome value in the
    # conversation_type slot. Should map to 'other', not raise.
    valid_leaf_dict["conversation_type"] = "informational"
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.conversation_type == "other"


def test_outcome_crosswire_falls_back(valid_leaf_dict):
    # Mirror: ConversationType value in the outcome slot.
    valid_leaf_dict["outcome"] = "planning"
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.outcome == "other"


def test_leaf_rejects_unknown_privacy_flag(valid_leaf_dict):
    # privacy_flags has no 'other' escape hatch and no fallback validator --
    # unknown values still hard-fail.
    valid_leaf_dict["privacy_flags"] = ["something-weird"]
    with pytest.raises(ValidationError):
        LeafSummary.model_validate(valid_leaf_dict)


# ---------------------------------------------------------------------------
# Topic normalization (mode="before" validator on LeafSummary.topics)
# ---------------------------------------------------------------------------

def test_topics_normalizes_uppercase_with_spaces(valid_leaf_dict):
    # Used to raise (topics had to be kebab-shaped on arrival). Since the
    # mode="before" normalizer landed, malformed-but-recoverable inputs are
    # rewritten in-place instead of rejected.
    valid_leaf_dict["topics"] = ["Topic With Space", "ok-topic", "another"]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.topics == ["topic-with-space", "ok-topic", "another"]


def test_topics_normalizes_dots(valid_leaf_dict):
    valid_leaf_dict["topics"] = ["php8.x", "type-hints"]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.topics == ["php8x", "type-hints"]


def test_topics_normalizes_apostrophes_and_spaces(valid_leaf_dict):
    valid_leaf_dict["topics"] = ["dexter's-lab", "topic with spaces"]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.topics == ["dexters-lab", "topic-with-spaces"]


def test_topics_dedupes_after_normalize(valid_leaf_dict):
    # All three normalize to "php8x" -- different on input, same on output.
    valid_leaf_dict["topics"] = ["php8.x", "PHP8X", "Php8.X"]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.topics == ["php8x"]


def test_topics_drops_unsalvageable(valid_leaf_dict):
    valid_leaf_dict["topics"] = ["!!!", "real-topic"]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.topics == ["real-topic"]


# ---------------------------------------------------------------------------
# Negative: pattern violations
# ---------------------------------------------------------------------------

def test_natural_language_must_be_iso(valid_leaf_dict):
    valid_leaf_dict["natural_language"] = "english"
    with pytest.raises(ValidationError, match="natural_language"):
        LeafSummary.model_validate(valid_leaf_dict)


# ---------------------------------------------------------------------------
# tech_stack normalization (mode="before" validator)
# ---------------------------------------------------------------------------

def test_tech_stack_normalizes_uppercase(valid_leaf_dict):
    # Used to raise; now normalizes. The four recurring offenders observed
    # in production: 'cPanel', 'mod_security', 'pg_moonshot', 'Has Spaces'.
    valid_leaf_dict["tech_stack"] = ["Has Spaces"]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.tech_stack == ["has-spaces"]


def test_tech_stack_normalizes_underscores(valid_leaf_dict):
    valid_leaf_dict["tech_stack"] = ["pg_moonshot", "mod_security"]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.tech_stack == ["pg-moonshot", "mod-security"]


def test_tech_stack_normalizes_camelcase(valid_leaf_dict):
    valid_leaf_dict["tech_stack"] = ["cPanel"]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.tech_stack == ["cpanel"]


def test_tech_stack_preserves_dot_plus_slash(valid_leaf_dict):
    # _KEBAB_DOT charset preserves legitimate identifiers that contain dots,
    # pluses, or slashes -- these carry semantic meaning in tech identifiers.
    valid_leaf_dict["tech_stack"] = ["php8.5", "c++", "production/staging"]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.tech_stack == ["php8.5", "c++", "production/staging"]


def test_tech_stack_dedupes_after_normalize(valid_leaf_dict):
    valid_leaf_dict["tech_stack"] = ["cPanel", "CPANEL", "cpanel"]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.tech_stack == ["cpanel"]


# ---------------------------------------------------------------------------
# Negative: list length / required field constraints
# ---------------------------------------------------------------------------

def test_topics_min_relaxed(valid_leaf_dict):
    """Floor was demoted to a soft signal; few-topic payloads now validate."""
    valid_leaf_dict["topics"] = ["one", "two"]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.topics == ["one", "two"]


def test_topics_max_relaxed(valid_leaf_dict):
    """Max-length removed (soft cap only). Model can over-deliver without retry."""
    valid_leaf_dict["topics"] = [f"t-{i}" for i in range(15)]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert len(m.topics) == 15


def test_semantic_keywords_min_relaxed(valid_leaf_dict):
    """Floor on semantic_keywords was demoted; small lists now validate."""
    valid_leaf_dict["semantic_keywords"] = ["one", "two"]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.semantic_keywords == ["one", "two"]


def test_title_min_length(valid_leaf_dict):
    valid_leaf_dict["title"] = "abc"
    with pytest.raises(ValidationError, match="title"):
        LeafSummary.model_validate(valid_leaf_dict)


def test_turn_count_must_be_positive(valid_leaf_dict):
    valid_leaf_dict["turn_count"] = 0
    with pytest.raises(ValidationError, match="turn_count"):
        LeafSummary.model_validate(valid_leaf_dict)


# ---------------------------------------------------------------------------
# Negative: shape mismatches (the bug we hit in production: string-where-array)
# ---------------------------------------------------------------------------

def test_topics_must_be_list_not_string(valid_leaf_dict):
    valid_leaf_dict["topics"] = "single,string,instead"
    with pytest.raises(ValidationError):
        LeafSummary.model_validate(valid_leaf_dict)


def test_key_points_must_be_list_not_string(valid_leaf_dict):
    valid_leaf_dict["key_points"] = "<parameter>foo</parameter>"
    with pytest.raises(ValidationError):
        LeafSummary.model_validate(valid_leaf_dict)


def test_extra_fields_forbidden(valid_leaf_dict):
    valid_leaf_dict["unknown_field"] = "anything"
    with pytest.raises(ValidationError, match="unknown_field|extra"):
        LeafSummary.model_validate(valid_leaf_dict)

# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_all_optional_arrays_can_be_empty(valid_leaf_dict):
    """Empty arrays for outputs / artifacts / citations / etc. are fine."""
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.outputs == []
    assert m.artifacts == []
    assert m.citations == []


def test_leaf_with_full_optionals(valid_leaf_dict):
    valid_leaf_dict["citations"] = [{"type": "url", "ref": "https://x", "title": "X"}]
    valid_leaf_dict["concepts_introduced"] = [{"name": "Foo", "brief": "bar"}]
    valid_leaf_dict["action_items"] = ["do thing"]
    valid_leaf_dict["privacy_flags"] = ["pii", "credentials"]
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.citations[0].type == "url"
    assert m.concepts_introduced[0].name == "Foo"


def test_natural_language_locale_form(valid_leaf_dict):
    valid_leaf_dict["natural_language"] = "en-US"
    m = LeafSummary.model_validate(valid_leaf_dict)
    assert m.natural_language == "en-US"


def test_root_time_distribution_pattern():
    from reindex.models import TimeBucket
    with pytest.raises(ValidationError):
        TimeBucket.model_validate({"year_month": "08-2024", "count": 1})  # wrong order


def test_citation_requires_type_and_ref():
    from reindex.models import Citation
    with pytest.raises(ValidationError):
        Citation.model_validate({"ref": "https://x"})  # missing type
    with pytest.raises(ValidationError):
        Citation.model_validate({"type": "url"})  # missing ref
