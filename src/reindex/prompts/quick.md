You are maintaining the README.md and metadata for the claude.ai export at {{EXPORT_DIR}}.

Layout:
  - conversations/<slug>/  one folder per standalone conversation
  - projects/<slug>/       one folder per project (knowledge + conversations)

Work through these steps explicitly, narrating what you're doing at each step:

1. Count entries:
   - Run `ls {{EXPORT_DIR}}/conversations | wc -l` to get conversation count.
   - Run `ls {{EXPORT_DIR}}/projects | wc -l` to get project count.
   - Run `ls {{EXPORT_DIR}}/projects` to capture the project slugs as a list.
   - Run `date -u +%Y-%m-%dT%H:%M:%SZ` for the last_sync timestamp.

2. Read the existing README.md and METADATA.json (if present) so you can diff against the new values rather than blindly overwriting.

3. Write or update README.md at the repo root with:
   - one-paragraph description of what this directory is
   - current totals (conversation count, project count, last sync timestamp)
   - how to re-sync: 'claudesync export-all --output <this directory>'
   - how to refresh metadata only: 'csindex quick'
   - brief layout map

4. Write or update METADATA.json at the repo root with:
   {
     "last_sync": "<UTC ISO-8601>",
     "conversation_count": <int>,
     "project_count": <int>,
     "projects": ["<slug>", ...]
   }

5. After writing, re-read both files and confirm the values match what step 1 produced. If anything is off, fix it.

Constraints:
- Do not modify anything inside conversations/ or projects/.
- Be terse. No marketing copy. No emojis.
- Sort the projects list alphabetically.
