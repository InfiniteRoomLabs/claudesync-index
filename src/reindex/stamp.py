"""
Stamp authoritative values into a markdown file's YAML frontmatter,
overriding whatever the LLM wrote. Used for cache-key fields the LLM
can't be trusted to copy verbatim (long hex hashes).
"""

from __future__ import annotations

import re
from pathlib import Path

# claudesync writes `- **Model:** <model>` into each conversation README.md.
_CONVERSATION_MODEL_RE = re.compile(r"^- \*\*Model:\*\* (.+)$", re.MULTILINE)


def read_conversation_model(conv_dir: Path) -> str:
    """Extract the model the *conversation* ran on from its README.md.

    Distinct from the `model` frontmatter field, which records the summarizer
    model. The conversation model is upstream mirror data (claudesync writes it
    as `- **Model:** <model>` in the conversation README), so we read it from
    disk rather than asking the LLM. Returns 'unknown' when the README or the
    line is absent. Shared by the leaf finalizer and the repair-hashes backfill.
    """
    readme = conv_dir / "README.md"
    try:
        text = readme.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unknown"
    m = _CONVERSATION_MODEL_RE.search(text)
    return m.group(1).strip() if m else "unknown"


def stamp_frontmatter(md_file: Path, field: str, value: str) -> None:
    """
    Updates or inserts a frontmatter field in a Markdown file.

    This function modifies a Markdown file to either add or update a frontmatter
    field. If the file does not contain frontmatter, a new block is prepended to
    the file. If the field already exists in the frontmatter, its value is
    updated. Otherwise, a new line for the field is added to the frontmatter.

    Parameters:
        md_file (Path): A Path object representing the Markdown file to modify.
        field (str): The field name to add or update in the frontmatter.
        value (str): The value to set for the specified field.

    Returns:
        None
    """
    text = md_file.read_text(encoding="utf-8")
    if not text.startswith("---"):
        # No frontmatter at all — prepend a new block.
        new = f"---\n{field}: {value}\n---\n\n{text}"
        md_file.write_text(new, encoding="utf-8")
        return

    end = text.find("\n---", 4)
    if end == -1:
        return

    fm_lines = text[4:end].splitlines()
    body = text[end:]
    prefix = f"{field}:"
    replaced = False
    for i, line in enumerate(fm_lines):
        if line.startswith(prefix):
            fm_lines[i] = f"{field}: {value}"
            replaced = True
            break
    if not replaced:
        fm_lines.insert(0, f"{field}: {value}")

    new_fm = "\n".join(fm_lines)
    md_file.write_text(f"---\n{new_fm}{body}", encoding="utf-8")


def stamp_many(md_file: Path, fields: dict[str, str]) -> None:
    """
    Updates the frontmatter of a markdown file with multiple key-value pairs.

    Loops through the provided dictionary of fields and stamps each key-value
    pair onto the frontmatter of the specified markdown file.

    Parameters:
    md_file: Path
        Path to the markdown file whose frontmatter will be updated.
    fields: dict[str, str]
        A dictionary containing key-value pairs to be added to the frontmatter.

    Returns:
    None
    """
    for k, v in fields.items():
        stamp_frontmatter(md_file, k, v)
