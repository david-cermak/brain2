# harvest — esp-modem Knowledge Base Pipeline

Builds a knowledge base from GitHub issues related to **esp-modem**, PPP, CMUX,
and cellular modem integration on ESP32.

## Pipeline Overview

```
[1. harvester.py]  →  [2. analyzer.py]  →  [2.5 graph_builder.py]  →  [3. indexer (future)]
 GitHub API            OpenAI LLM           Merge & build KG          Supabase pgvector
 ↓                     ↓                    ↓                         ↓
 harvest.db            harvest.db           harvest.db                vector search
 (issues table)        (analysis table)     (kg_entities/relations)
                       (kg_raw_* tables)    knowledge/knowledge_graph.json
                       knowledge/issues/*   knowledge/knowledge_graph.graphml
```

## Setup

```bash
pip install PyGithub python-dotenv openai
```

Create a `.env` file in this directory:

```
GITHUB_TOKEN=ghp_...
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://litellm.espressif.tools
OPENAI_MODEL=gpt-5.2
```

## Stage 1: Harvest (`harvester.py`)

Scans GitHub repos for issues matching modem-related keywords and stores them
in a local SQLite database (`harvest.db`).

**Repos scanned:** `espressif/esp-idf`, `espressif/esp-protocols`

**Keywords (regex, case-insensitive):** PPP, PPPoS, esp-modem, esp_modem,
AT command, DCE, DTE, modem, CMUX, ppp_netif, esp_netif_ppp

**How it works:**
- Fetches all issues (open + closed) from each repo
- Quick keyword filter on title + body + labels (no extra API call)
- If matched, fetches comments and stores everything in SQLite
- Incremental: compares `updated_at` timestamps, skips unchanged issues on re-run

```bash
# Harvest all repos (esp-idf is slow — ~15k issues)
python harvester.py

# Or run from Python to harvest a single repo:
python -c "
from harvester import *
from github import Github
g = Github(GITHUB_TOKEN, per_page=100)
conn = init_db()
harvest_repo(g, 'espressif/esp-protocols', conn)
conn.close()
"
```

**Database schema (`issues` table):**

| Column         | Type    | Description                              |
|----------------|---------|------------------------------------------|
| repo           | TEXT    | e.g. `espressif/esp-protocols`           |
| issue_number   | INTEGER | GitHub issue number                      |
| title          | TEXT    | Issue title                              |
| body           | TEXT    | Issue body (markdown)                    |
| labels         | TEXT    | Comma-separated label names              |
| state          | TEXT    | `open` or `closed`                       |
| html_url       | TEXT    | Link to the issue on GitHub              |
| user_login     | TEXT    | Author's GitHub username                 |
| comments_json  | TEXT    | JSON array of `{user, body, created_at}` |
| created_at     | TEXT    | ISO timestamp                            |
| updated_at     | TEXT    | ISO timestamp (used for incremental)     |
| fetched_at     | TEXT    | When we last fetched this issue          |
| processed_at   | TEXT    | NULL until analyzed by Stage 2           |

## Stage 2: Analyze (`analyzer.py`)

Sends unprocessed issues to an OpenAI-compatible LLM for structured analysis.

**For each issue, the LLM returns:**
- `relevant` — is this actually about esp-modem?
- `summary` — one paragraph describing the problem
- `root_cause` — what went wrong
- `solution` — how it was resolved
- `lessons` — actionable takeaways for developers
- `tags` — topic labels (e.g. `ppp`, `cmux`, `usb modem`)

**Output:**
- Results stored in `analysis` table in `harvest.db`
- Markdown file per relevant issue written to `../knowledge/issues/`

```bash
# Analyze all unprocessed issues
python analyzer.py

# Analyze only esp-protocols issues (faster for testing)
python analyzer.py --repo esp-protocols

# Analyze 10 issues as a quick test
python analyzer.py --repo esp-protocols --limit 10
```

**CLI arguments:**

| Flag      | Description                                             |
|-----------|---------------------------------------------------------|
| `--limit` | Max issues to process (default: all unprocessed)        |
| `--repo`  | Filter by repo substring, e.g. `esp-protocols`         |
| `--db`    | Path to harvest.db (default: `./harvest.db`)            |

## Output: Knowledge Files

Each relevant issue produces a markdown file in `knowledge/issues/`:

```
knowledge/issues/
  esp-protocols-33.md    # CMUX failure with BG96
  esp-protocols-44.md    # PPP retry/reconnect stalling
  esp-protocols-46.md    # CMUX 2-byte length encoding bug
  esp-idf-12345.md       # ...
```

Each file contains: summary, root cause, solution, lessons learned, and tags.

## Stage 2.5: Knowledge Graph (`graph_builder.py`)

Reads per-issue entity/relation extractions from `kg_raw_entities` and
`kg_raw_relations` tables, merges entities by normalized name, aggregates
relations, and produces a unified knowledge graph.

**How entities are extracted:**
- **New issues** (via `analyzer.py`): The LLM prompt now includes KG extraction
  alongside the existing analysis — entities and relations come back in the same
  JSON response at zero additional LLM cost.
- **Already-analyzed issues** (via `analyzer.py --extract-kg`): A lightweight
  KG-only prompt is sent using the condensed analysis text (summary, root cause,
  solution, lessons) — much cheaper than re-analyzing the full issue thread.

**Entity types:** `modem_model`, `component`, `error_pattern`, `config_option`,
`symptom`, `root_cause_cat`, `solution_pattern`, `idf_version`

**Relation types:** `EXHIBITS`, `CAUSED_BY`, `FIXED_BY`, `INVOLVES_CONFIG`,
`ORIGINATES_FROM`, `AFFECTS_COMPONENT`, `HAS_QUIRK`, `VERSION_SPECIFIC`

```bash
# Extract KG from already-analyzed issues (lightweight KG-only LLM call)
python analyzer.py --extract-kg

# Extract KG for specific repo only, limit to 10 issues
python analyzer.py --extract-kg --repo esp-protocols --limit 10

# Build the merged knowledge graph
python graph_builder.py
```

**Output:**
- Merged `kg_entities` and `kg_relations` tables in `harvest.db`
- `knowledge/knowledge_graph.json` — full graph with stats
- `knowledge/knowledge_graph.graphml` — for visualization (yEd, Gephi, etc.)

**Database tables:**

| Table | Description |
|-------|-------------|
| `kg_raw_entities` | Per-issue entity extractions (raw LLM output) |
| `kg_raw_relations` | Per-issue relation extractions (raw LLM output) |
| `kg_entities` | Merged entities (built by `graph_builder.py`) |
| `kg_relations` | Merged relations (built by `graph_builder.py`) |

**Example graph traversal:**
```
[modem_model] BG96
  --EXHIBITS--> [symptom] cmux setup failure
    --CAUSED_BY--> [root_cause_cat] strict UIH type assumption
      --FIXED_BY--> [solution_pattern] tolerant UIH parsing
```

## Stage 3: Index (future)

Will embed the summaries + lessons into Supabase pgvector for RAG-style
semantic search. See `../knowledge/search.py` for the existing prototype.

## Re-running / Incremental Updates

- **Harvester:** Safe to re-run. Skips issues whose `updated_at` hasn't changed.
  If a new comment is added to an old issue, it gets re-fetched.
- **Analyzer:** Only processes issues where `processed_at IS NULL`. To re-analyze
  an issue, set its `processed_at` back to NULL in the DB.
- **KG extraction:** `--extract-kg` only processes analyzed issues that don't
  have KG data yet. To re-extract, delete rows from `kg_raw_entities` for that issue.
- **Graph builder:** Safe to re-run. Drops and rebuilds the merged tables each time.
