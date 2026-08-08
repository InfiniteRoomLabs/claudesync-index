"""
Auto-coercion middleware for LLM tool-use outputs that don't quite match schema.

Anthropic's tool input_schema is advisory only — the model returns whatever it
wants. Many violations are mechanical (string-where-array, null-where-list,
unknown extra keys, sentinel dates) and can be fixed without re-prompting.

This module pre-processes tool_input dicts BEFORE Pydantic validation. If a fix
turns a malformed payload into a valid one, we save a full retry round-trip.

Tradeoffs: we err toward LIGHT TOUCH coercion. We don't invent missing required
data — fields that are genuinely absent still fail validation and trigger
retry. We just normalize shape mismatches the model commonly produces.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

# Single element that looks like CSV of terse tags (no sentence punctuation, no
# spaces inside individual tokens). Conservative — only triggers on tag-like
# content, not on free-text fields that may legitimately contain commas.
_COMMA_TAG_RE = re.compile(
    r"^\s*[a-z0-9][a-z0-9_\-\. ]*(\s*,\s*[a-z0-9][a-z0-9_\-\. ]*)+\s*$",
    re.IGNORECASE,
)

# Only fields where CSV-splitting is semantically safe.
_CSV_SAFE_FIELDS = frozenset({
    "topics", "semantic_keywords", "tech_stack", "code_languages",
    "entities", "action_items", "key_points",
})


def _looks_csv(value: str) -> bool:
    if not _COMMA_TAG_RE.match(value):
        return False
    if "." in value:  # likely sentence text
        return False
    if ":" in value:  # likely a structured value
        return False
    return True


def _maybe_csv_split(field_name: str, items: list) -> list:
    """If the only element is a CSV-looking string in a CSV-safe field, split it."""
    if field_name not in _CSV_SAFE_FIELDS:
        return items
    if len(items) != 1 or not isinstance(items[0], str):
        return items
    if not _looks_csv(items[0]):
        return items
    return [tok.strip() for tok in items[0].split(",") if tok.strip()]


def coerce_for_model(payload: Any, model_cls: type[BaseModel]) -> Any:
    """Best-effort fix-up of a payload dict against a Pydantic model's schema.

    - String where list[str] expected → wrap as [string].
    - null where list expected → [].
    - Single-element list with a CSV-looking string in a tag-like field → split.
    - Unknown extra keys → drop (when the model has extra='forbid').
    - Recurse into nested models.

    Returns a new dict; does not mutate input.
    """
    if not isinstance(payload, dict):
        return payload

    fields = model_cls.model_fields
    out: dict[str, Any] = {}

    # Drop unknown keys when extra='forbid'.
    extra_forbidden = (
        getattr(model_cls, "model_config", {}).get("extra") == "forbid"
    )

    for key, value in payload.items():
        if key not in fields:
            if extra_forbidden:
                continue
            out[key] = value
            continue

        field_info = fields[key]
        coerced = _coerce_field(value, field_info.annotation)
        # CSV-split layer: only for list-typed results in CSV-safe fields.
        if isinstance(coerced, list):
            coerced = _maybe_csv_split(key, coerced)
        out[key] = coerced

    return out


def _coerce_field(value: Any, annotation: Any) -> Any:
    """Coerce a single field value toward its declared annotation."""
    origin = _get_origin(annotation)
    args = _get_args(annotation)

    # list[X] field
    if origin in (list, "list"):
        if value is None:
            return []
        if isinstance(value, str):
            # String where array expected — wrap as single-element list.
            # Empty string -> empty list.
            return [value] if value else []
        if isinstance(value, list):
            inner = args[0] if args else Any
            return [_coerce_field(item, inner) for item in value]
        # Anything else (int, dict, etc.) — wrap.
        return [value]

    # Union types (e.g. str | None or Optional[X])
    if origin in ("Union", "UnionType") or _is_union(annotation):
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce_field(value, arg)
            except Exception:
                continue
        return value

    # Nested BaseModel
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if isinstance(value, dict):
            return coerce_for_model(value, annotation)
        return value

    # Scalar — return as-is (Pydantic handles type coercion for primitives).
    return value


def _get_origin(annotation: Any) -> Any:
    """Cross-Python typing.get_origin() with stringy fallback for forward refs."""
    import typing
    origin = typing.get_origin(annotation)
    if origin is not None:
        return origin
    return None


def _get_args(annotation: Any) -> tuple:
    import typing
    return typing.get_args(annotation)


def _is_union(annotation: Any) -> bool:
    import types
    import typing
    origin = typing.get_origin(annotation)
    return origin is typing.Union or (hasattr(types, "UnionType") and origin is types.UnionType)
