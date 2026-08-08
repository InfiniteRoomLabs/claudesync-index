"""
Pydantic models = single source of truth for index step shapes.

Adding a frontmatter or body field = add an attribute here, update render.py.
The Anthropic tool_use input_schema is generated automatically from these models
via .model_json_schema(); no separate schema files to maintain.

Stamped fields (slug, type, content_hash/children_hash/inputs_hash, generated_at,
model) are NOT in these models — they are written by stamp.py post-render.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Shared validators
# ---------------------------------------------------------------------------

_KEBAB = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_KEBAB_DOT = re.compile(r"^[a-z0-9][a-z0-9.+/-]*$")
_LANG = re.compile(r"^[a-z0-9][a-z0-9+-]*$")

# Drop chars that are noise inside a token vs. chars that mark word boundaries
_KEBAB_WORD_SEP = re.compile(r"[\s_/]+")  # boundaries -> hyphen
_KEBAB_DROP = re.compile(r"[^a-z0-9-]+")  # noise -> nothing
_KEBAB_DASH_RUN = re.compile(r"-{2,}")

def _normalize_kebab(s: str) -> str:
    """Coerce a model-emitted tag into kebab-case, or '' if nothing survives.

    - lowercase
    - whitespace, underscore, slash -> single hyphen (preserve word boundary)
    - everything else outside of [a-z0-9-] dropped (apostrophes, dots, etc.)
    - collapse runs of hyphens; trim leading/trailing
    """
    s = s.lower()
    s = _KEBAB_WORD_SEP.sub("-", s)
    s = _KEBAB_DROP.sub("", s)
    s = _KEBAB_DASH_RUN.sub("-", s).strip("-")
    return s


# tech_stack accepts a wider charset than topics: dots (`php8.5`), pluses
# (`c++`), slashes (`production/staging`) all carry meaning. Only normalize
# what's clearly out-of-spec: uppercase + underscore + whitespace. Slashes
# are PRESERVED here (unlike topic normalization where they collapse to
# hyphens) because `production/staging` is a real and useful identifier.
_KEBAB_DOT_WORD_SEP = re.compile(r"[\s_]+")
_KEBAB_DOT_DROP = re.compile(r"[^a-z0-9.+/\-]+")


def _normalize_kebab_dot(s: str) -> str:
    """Coerce a tech_stack identifier into the _KEBAB_DOT shape, or '' if empty.

    - lowercase
    - whitespace, underscore -> single hyphen (word boundary; slash kept)
    - everything else outside [a-z0-9.+/-] dropped
    - collapse runs of hyphens; trim leading/trailing
    """
    s = s.lower()
    s = _KEBAB_DOT_WORD_SEP.sub("-", s)
    s = _KEBAB_DOT_DROP.sub("", s)
    s = _KEBAB_DASH_RUN.sub("-", s).strip("-")
    return s


# ---------------------------------------------------------------------------
# Leaf
# ---------------------------------------------------------------------------

CitationType = Literal["url", "paper", "book", "rfc", "issue", "other"]


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: CitationType
    ref: str
    title: str | None = None


class Concept(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    brief: str = Field(..., description="One-sentence definition.")


ConversationType = Literal[
    "how-to", "debug", "brainstorm", "code-review", "research",
    "planning", "learning", "venting", "reference-lookup",
    "decision", "exploration", "other",  # 'other' as escape hatch
]
Outcome = Literal[
    "resolved", "partial", "abandoned", "exploratory", "ongoing",
    "informational", "other",
]
Complexity = Literal["trivial", "simple", "moderate", "deep"]
Reusability = Literal["high", "medium", "low"]
PrivacyFlag = Literal[
    "pii", "credentials", "company-confidential",
    "third-party-confidential", "medical", "financial",
]

# Pre-baked allowed-values sets for fields with `mode="before"` enum-fallback
# coercion. Pulling from get_args() once at import keeps the validator
# allocation-free at runtime and keeps the source of truth on the Literal
# itself.
_ENUM_ALLOWED_BY_FIELD: dict[str, frozenset[str]] = {
    "conversation_type": frozenset(get_args(ConversationType)),
    "outcome": frozenset(get_args(Outcome)),
}

_DATE_SENTINELS = {"<unknown>", "unknown", "n/a", "na", "none", "null", "", "tbd"}
_FALLBACK_DATE = date(1970, 1, 1)


def _coerce_date(v: Any) -> Any:
    """Map sentinels like '<UNKNOWN>' / 'n/a' to the epoch fallback so ungated
    LLM responses validate. Real ISO dates and date instances pass through."""
    if isinstance(v, str) and v.strip().lower() in _DATE_SENTINELS:
        return _FALLBACK_DATE
    return v


class LeafSummary(BaseModel):
    """Per-conversation index payload.

    Tightness philosophy: floors stay (min_length on must-have arrays;
    required fields stay required), but ceilings drop (max_length removed)
    because the model treats those as guidelines, and we don't want to retry
    over a 31st semantic keyword. Optional arrays default to empty, so a model
    that omits, e.g. citations validates without a retry.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=4, max_length=200, description="Short human title, 4-10 words.")
    summary: str = Field(..., description="3-6 sentence narrative. Concrete, no filler.")
    embedding_text: str = Field(..., description="1-2 paragraph dense summary tuned for vector embedding.")

    topics: list[str] = Field(
        ...,
        description=(
            "Aim for 3-8 lowercase-hyphenated tags. Soft target — pipeline accepts fewer "
            'and logs a quality signal. Example: ["postgres","async","sqlalchemy"].'
        ),
    )
    semantic_keywords: list[str] = Field(
        ...,
        description=(
            "Aim for 5-30 single words/short phrases for sparse retrieval. Soft target — "
            "pipeline accepts fewer and logs a quality signal."
        ),
    )

    key_points: list[str] = Field(..., min_length=1)
    # Optional arrays default to [] so a model that omits them validates.
    outputs: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)

    turn_count: int = Field(..., ge=1)
    date_range_start: date
    date_range_end: date

    conversation_type: ConversationType
    outcome: Outcome
    complexity: Complexity
    reusability: Reusability

    tech_stack: list[str] = Field(
        default_factory=list,
        description="Normalized identifiers of frameworks, libraries, tools, services."
    )
    code_languages: list[str] = Field(
        default_factory=list,
        description="Lowercase language identifiers; empty if no code."
    )
    has_code: bool

    entities: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    concepts_introduced: list[Concept] = Field(default_factory=list)

    action_items: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)

    privacy_flags: list[PrivacyFlag] = Field(default_factory=list)
    natural_language: str = Field(..., description="ISO 639-1 code (en, zh, en-US).")

    @field_validator("date_range_start", "date_range_end", mode="before")
    @classmethod
    def _date_fallback(cls, v: Any) -> Any:
        return _coerce_date(v)

    @field_validator("topics", "code_languages")
    @classmethod
    def _kebab(cls, v: list[str]) -> list[str]:
        for s in v:
            if not _KEBAB.match(s) and not _LANG.match(s):
                raise ValueError(f"not lowercase-hyphenated: {s!r}")
        return v

    @field_validator("tech_stack")
    @classmethod
    def _kebab_dot(cls, v: list[str]) -> list[str]:
        for s in v:
            if not _KEBAB_DOT.match(s):
                raise ValueError(f"not normalized identifier: {s!r}")
        return v

    @field_validator("natural_language")
    @classmethod
    def _iso639(cls, v: str) -> str:
        if not re.match(r"^[a-z]{2}(-[A-Z]{2})?$", v):
            raise ValueError(f"not ISO 639-1: {v!r}")
        return v

    @field_validator("topics", mode="before")
    @classmethod
    def _normalize_topics(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v # let the type check downstream report the shape error
        out: list[str] = []
        seen: set[str] = set()
        for raw in v:
            if not isinstance(raw, str):
                return v # same - let downstream report
            norm = _normalize_kebab(raw)
            if norm and norm not in seen:
                seen.add(norm)
                out.append(norm)
        return out

    @field_validator("tech_stack", mode="before")
    @classmethod
    def _normalize_tech_stack(cls, v: Any) -> Any:
        """Mirror of _normalize_topics for tech_stack, using _KEBAB_DOT charset.

        Catches the recurring failures: `cPanel`, `mod_security`, `pg_moonshot`
        -- uppercase and underscore -- without touching legitimate identifiers
        like `php8.5`, `c++`, or `production/staging`.
        """
        if not isinstance(v, list):
            return v
        out: list[str] = []
        seen: set[str] = set()
        for raw in v:
            if not isinstance(raw, str):
                return v
            norm = _normalize_kebab_dot(raw)
            if norm and norm not in seen:
                seen.add(norm)
                out.append(norm)
        return out

    @field_validator("conversation_type", "outcome", mode="before")
    @classmethod
    def _enum_fallback(cls, v: Any, info: Any) -> Any:
        """Map unrecognized enum values to 'other' instead of failing.

        Observed model confusion: emits an `Outcome` value in the
        `ConversationType` slot and vice versa (e.g. `conversation_type=
        'informational'`, `outcome='planning'`). Both literals include
        `'other'` as an explicit escape hatch -- use it rather than
        retrying. Caller-visible signal lives in the debug log; the
        original value is included so we can see model confusion when we
        review the run.
        """
        if not isinstance(v, str):
            return v
        allowed = _ENUM_ALLOWED_BY_FIELD.get(info.field_name)
        if allowed is None or v in allowed:
            return v
        return "other"


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

ProjectStatus = Literal["active", "dormant", "archived", "shipped", "abandoned"]
Velocity = Literal["accelerating", "steady", "declining", "dormant"]
DominantOutcome = Literal["resolved", "partial", "abandoned", "exploratory", "ongoing", "mixed"]


class ConversationRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slug: str
    title: str
    gist: str


class KnowledgeFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str
    description: str


class TechCount(BaseModel):
    # ge=0 because aggregator may emit a 0-count for techs newly added but not yet
    # observed in any child. Keep loose; semantics handled by display ordering.
    model_config = ConfigDict(extra="forbid")
    name: str
    count: int = Field(..., ge=0)


class OpenActionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    from_slug: str
    item: str


class ProjectAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    embedding_text: str

    conversations: list[ConversationRef] = Field(default_factory=list)
    knowledge_files: list[KnowledgeFile] = Field(default_factory=list)
    recurring_themes: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)

    conversation_count: int = Field(..., ge=0)
    knowledge_count: int = Field(..., ge=0)
    date_range_start: date
    date_range_end: date

    project_status: ProjectStatus
    velocity: Velocity
    dominant_outcome: DominantOutcome

    tech_stack: list[TechCount] = Field(default_factory=list)
    open_action_items: list[OpenActionItem] = Field(default_factory=list)

    @field_validator("date_range_start", "date_range_end", mode="before")
    @classmethod
    def _date_fallback(cls, v: Any) -> Any:
        return _coerce_date(v)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

class ProjectRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slug: str
    gist: str


class TimeBucket(BaseModel):
    model_config = ConfigDict(extra="forbid")
    year_month: str = Field(..., pattern=r"^[0-9]{4}-[0-9]{2}$")
    count: int = Field(..., ge=0)


class EntityCount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    count: int = Field(..., ge=1)


class CitationCount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ref: str
    title: str | None = None
    count: int = Field(..., ge=1)


class TechTimeline(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tech: str
    first_seen: date
    last_seen: date
    count: int = Field(..., ge=1)


class KnowledgeCluster(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    sample_topics: list[str]
    conversation_count: int = Field(..., ge=1)


class RootAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overview: str
    embedding_text: str

    projects: list[ProjectRef] = Field(default_factory=list)
    top_themes: list[str] = Field(default_factory=list)
    standalone_overview: str
    top_topics: list[str] = Field(default_factory=list)

    project_count: int = Field(..., ge=0)
    conversation_count: int = Field(..., ge=0)
    date_range_start: date
    date_range_end: date

    time_distribution: list[TimeBucket] = Field(default_factory=list)
    top_entities: list[EntityCount] = Field(default_factory=list)
    top_citations: list[CitationCount] = Field(default_factory=list)
    tech_stack_timeline: list[TechTimeline] = Field(default_factory=list)
    knowledge_clusters: list[KnowledgeCluster] = Field(default_factory=list)

    @field_validator("date_range_start", "date_range_end", mode="before")
    @classmethod
    def _date_fallback(cls, v: Any) -> Any:
        return _coerce_date(v)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STEP_MODEL: dict[str, type[BaseModel]] = {
    "leaf": LeafSummary,
    "project": ProjectAggregate,
    "root": RootAggregate,
}


def schema_for(step: str) -> dict:
    """Return JSON Schema (draft-7) for a step. Used to build Anthropic tool input_schema."""
    return STEP_MODEL[step].model_json_schema()


def schema_cls_for(step: str) -> type[BaseModel]:
    """Return the Pydantic model class for a step. Used by 'work finalize' for validation."""
    return STEP_MODEL[step]
