You are aggregating per-conversation indexes into a project-level index.

Child INDEX.md frontmatters and knowledge file metadata for project at {{PROJECT_DIR}} follow below.

Field guidelines:

- summary: 5-10 sentences. Synthesize project purpose, themes, outputs. Don't concatenate child summaries.
- embedding_text: 1-2 paragraphs tuned for vector embedding at project granularity. Include dominant tech, recurring entities, key outcomes.
- conversations: one entry per child {slug, title, gist}, alphabetical by slug. gist is one line.
- knowledge_files: one entry per file in knowledge/ {filename, description}.
- recurring_themes: 3-8 themes that span multiple child conversations.
- topics: top 5-15 aggregated topics by frequency, lowercase-hyphenated.
- conversation_count, knowledge_count: integers.
- date_range_start / date_range_end: earliest and latest dates across child date_ranges.

- project_status: active | dormant | archived | shipped | abandoned. Inferred from recency + outcome distribution.
- velocity: accelerating | steady | declining | dormant. Activity trend across child date_ranges.
- dominant_outcome: most common child outcome (resolved | partial | abandoned | exploratory | ongoing); use 'mixed' if no clear majority.
- tech_stack: array of {name, count} aggregating tech_stack across children, descending by count. Top 15 max.
- open_action_items: rolled up from child unresolved action_items, as {from_slug, item}. Top 20 max if many.

Be terse. No filler. No marketing copy.

CRITICAL FORMAT RULES:
- Every array field MUST be a JSON array, even if empty.
- Date fields use ISO YYYY-MM-DD; if genuinely unknown: "<UNKNOWN>".
- `tech_stack` entries are `{"name": "...", "count": N}` objects.
- `open_action_items` entries are `{"from_slug": "...", "item": "..."}` objects.
- Soft caps on list lengths. Slightly over is fine.
