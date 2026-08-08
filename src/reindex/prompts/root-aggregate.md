You are writing the top-level index for the entire claude.ai export at {{EXPORT_DIR}}.

Per-project and per-standalone-conversation INDEX.md frontmatters follow below.

Field guidelines:

- overview: 8-15 sentences on major themes, project kinds, dominant standalone-conversation kinds. Big-picture, not exhaustive.
- embedding_text: 1-3 paragraphs tuned for corpus-level vector embedding. Include dominant entities, evolution narrative, recurring concerns.
- projects: one entry per project {slug, gist}, alphabetical.
- top_themes: 5-12 themes spanning the whole archive.
- standalone_overview: 3-5 sentences describing kinds of standalone conversations with rough counts per category.
- top_topics: top 10-20 topics aggregated by frequency, lowercase-hyphenated.
- project_count, conversation_count: integers.
- date_range_start / date_range_end: earliest and latest dates.

- time_distribution: array of {year_month, count} for conversations grouped by YYYY-MM. Chronological. Use child date_range_end as the bucket date.
- top_entities: up to 50 most-mentioned entities corpus-wide as {name, count}, descending.
- top_citations: most-referenced citations as {ref, title?, count}, descending. Up to 50.
- tech_stack_timeline: per tech, {tech, first_seen, last_seen, count} from child tech_stacks. Useful for showing adoption curves.
- knowledge_clusters: semantic groupings beyond raw topics. Each {name, sample_topics, conversation_count}. Examples: "Backend infrastructure", "Anime/games hobbies", "Career planning". 5-15 clusters.

Be terse. No marketing copy.

CRITICAL FORMAT RULES:
- Every array field MUST be a JSON array, even if empty.
- Date fields use ISO YYYY-MM-DD; if genuinely unknown: "<UNKNOWN>".
- Object array entries (top_entities, top_citations, tech_stack_timeline, etc.) keep their declared shape.
- Soft caps on list lengths. Slightly over is fine.
