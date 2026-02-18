#!/usr/bin/env python3
"""
Stage 2: LLM-based analysis of harvested esp-modem issues.

Reads unprocessed issues from harvest.db, sends each to the OpenAI API
for structured analysis, and stores results in an `analysis` table plus
individual markdown files under knowledge/issues/.
"""
import argparse
import json
import os
import sqlite3
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

DB_PATH = os.path.join(os.path.dirname(__file__), "harvest.db")
KNOWLEDGE_DIR = Path(os.path.dirname(__file__)).parent / "knowledge" / "issues"

SYSTEM_PROMPT = """\
You are an expert embedded-systems engineer specialising in ESP-IDF, \
esp-modem, PPP networking, and cellular modem integration.

You are building a knowledge base to help TRIAGE FUTURE esp-modem issues. \
Analyse the following GitHub issue thread and return a single JSON object \
with exactly these keys:

{
  "relevance": <int 0-100 — how useful this issue's learnings would be for \
triaging a FUTURE similar issue. See scoring guide below>,
  "relevance_reason": "<one sentence explaining the score>",
  "summary": "<one paragraph describing the problem>",
  "root_cause": "<what went wrong — null if unknown>",
  "solution": "<how it was resolved — null if unresolved or unknown>",
  "lessons": ["<actionable takeaway 1>", "..."],
  "tags": ["<topic tag>", "..."],
  "entities": [<see entity extraction rules below>],
  "relations": [<see relation extraction rules below>]
}

Relevance scoring guide (0-100):
  90-100: Issue reveals a non-obvious esp-modem/PPP/CMUX runtime bug, \
    modem-specific quirk, or protocol-level insight with a clear root cause \
    and fix that would directly help diagnose a similar future issue. \
    Example: CMUX frame type incompatibility with specific modem, DTR line \
    causing unexpected mode exit, DTE callback restoration bug.
  60-89: Issue is about esp-modem behavior with useful lessons, but the \
    root cause or solution is only partially clear, or the insight is \
    somewhat specific to one user's setup.
  30-59: Issue mentions esp-modem but the lessons are generic, the info is \
    inconclusive, or it has limited future triage value. Examples: user \
    integration mistakes, vague "it doesn't work" without clear resolution.
  0-29: NOT useful for future triage. Score low for ANY of these reasons:
    - General programming knowledge (e.g. virtual destructors, error handling)
    - Build / compile / linker / CMake / component-manager issues
    - Application bugs unrelated to modem library code
    - Already fixed trivially in the modem layer with no reoccurrence risk
    - Stack overflow or generic RTOS issues not specific to modem
    - Not enough information to draw actionable conclusions
    - Documentation or API signature mismatch issues
    - Feature requests or "how to" questions without technical depth

=== KNOWLEDGE GRAPH ENTITY/RELATION EXTRACTION ===

Extract typed entities and relations to build a knowledge graph for triage.

Entity types (use ONLY these):
  modem_model     — specific modem hardware, e.g. "SIM800", "A7670G", "BG96"
  component       — software component/layer, e.g. "esp_modem", "lwIP", "CMUX", "PPP"
  error_pattern   — specific error/log message, e.g. "Loopback detected", "NO CARRIER"
  config_option   — Kconfig/sdkconfig option, e.g. "ESP_MODEM_PPP_ESCAPE_BEFORE_EXIT"
  symptom         — observable problem, e.g. "reconnect failure", "mode switch timeout"
  root_cause_cat  — underlying technical cause category, e.g. "escape sequence race"
  solution_pattern — reusable fix approach, e.g. "reconnect loop COMMAND→DATA cycle"
  idf_version     — ESP-IDF version, e.g. "v5.5.2"

Relation types (use ONLY these):
  EXHIBITS           — modem_model → symptom
  CAUSED_BY          — symptom → root_cause_cat
  FIXED_BY           — root_cause_cat → solution_pattern
  INVOLVES_CONFIG    — solution_pattern → config_option
  ORIGINATES_FROM    — error_pattern → component (where the error is emitted)
  AFFECTS_COMPONENT  — symptom → component
  HAS_QUIRK          — modem_model → root_cause_cat (device-specific behavior)
  VERSION_SPECIFIC   — root_cause_cat → idf_version

Entity format: {"name": "<short identifier>", "type": "<type>", "description": "<one sentence>"}
Relation format: {"source": "<entity name>", "target": "<entity name>", "type": "<RELATION_TYPE>", "weight": <0.0-1.0>}

Entity/relation rules:
- Names should be SHORT identifiers: "SIM800" not "SIMCOM SIM800 modem module"
- Preserve original case for modem_model and config_option; lowercase for others
- Only extract entities clearly relevant to esp-modem domain
- weight: confidence/strength of the relation (1.0 = certain, 0.5 = plausible)
- If relevance < 30, set entities and relations to empty lists []
- Aim for 3-10 entities and 2-8 relations per issue (fewer is fine if the issue is simple)

Rules:
- Return ONLY valid JSON, no markdown fences, no commentary.
- "tags" should be short lowercase labels like "ppp", "cmux", "usb modem", \
"carrier loss", "at commands", "dce/dte", "netif", "ota", "memory leak", etc.
- "lessons" should be concrete and specific to esp-modem — NOT generic \
programming advice. If you can't write a lesson that's specific to modem/PPP \
behavior, that's a sign the relevance score should be lower.
- If relevance < 30, set lessons to an empty list [].
"""

KG_EXTRACT_PROMPT = """\
You are an expert embedded-systems engineer specialising in ESP-IDF, \
esp-modem, PPP networking, and cellular modem integration.

Given the analysis of a GitHub issue below, extract typed entities and \
relations for a knowledge graph. Return a JSON object with exactly these keys:

{
  "entities": [{"name": "<short identifier>", "type": "<type>", "description": "<one sentence>"}],
  "relations": [{"source": "<entity name>", "target": "<entity name>", "type": "<RELATION_TYPE>", "weight": <0.0-1.0>}]
}

Entity types (use ONLY these):
  modem_model, component, error_pattern, config_option, symptom, \
  root_cause_cat, solution_pattern, idf_version

Relation types (use ONLY these):
  EXHIBITS (modem_model → symptom), \
  CAUSED_BY (symptom → root_cause_cat), \
  FIXED_BY (root_cause_cat → solution_pattern), \
  INVOLVES_CONFIG (solution_pattern → config_option), \
  ORIGINATES_FROM (error_pattern → component), \
  AFFECTS_COMPONENT (symptom → component), \
  HAS_QUIRK (modem_model → root_cause_cat), \
  VERSION_SPECIFIC (root_cause_cat → idf_version)

Rules:
- Names should be SHORT identifiers: "SIM800" not "SIMCOM SIM800 modem module"
- Preserve original case for modem_model and config_option; lowercase for others
- Only extract entities clearly relevant to esp-modem domain
- weight: confidence/strength (1.0 = certain, 0.5 = plausible)
- Aim for 3-10 entities and 2-8 relations (fewer is fine for simple issues)
- Return ONLY valid JSON, no markdown fences, no commentary.
"""


# ── Database ──────────────────────────────────────────────────

VALID_ENTITY_TYPES = {
    "modem_model", "component", "error_pattern", "config_option",
    "symptom", "root_cause_cat", "solution_pattern", "idf_version",
}

VALID_RELATION_TYPES = {
    "EXHIBITS", "CAUSED_BY", "FIXED_BY", "INVOLVES_CONFIG",
    "ORIGINATES_FROM", "AFFECTS_COMPONENT", "HAS_QUIRK", "VERSION_SPECIFIC",
}


def init_analysis_table(conn: sqlite3.Connection) -> None:
    """Create or migrate the analysis table."""
    # Check if we need to migrate from the old schema (relevant BOOLEAN → relevance INTEGER)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(analysis)").fetchall()}
    if "relevant" in cols and "relevance" not in cols:
        print("Migrating analysis table: relevant → relevance ...", file=sys.stderr, flush=True)
        conn.execute("DROP TABLE analysis")
        conn.execute("UPDATE issues SET processed_at = NULL")
        conn.commit()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis (
            repo              TEXT    NOT NULL,
            issue_number      INTEGER NOT NULL,
            relevance         INTEGER,
            relevance_reason  TEXT,
            summary           TEXT,
            root_cause        TEXT,
            solution          TEXT,
            lessons_json      TEXT,
            tags_json         TEXT,
            analyzed_at       TEXT,
            PRIMARY KEY (repo, issue_number)
        )
    """)
    conn.commit()


def init_kg_tables(conn: sqlite3.Connection) -> None:
    """Create the KG raw extraction tables (per-issue entities and relations)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kg_raw_entities (
            repo            TEXT    NOT NULL,
            issue_number    INTEGER NOT NULL,
            name            TEXT    NOT NULL,
            type            TEXT    NOT NULL,
            description     TEXT,
            extracted_at    TEXT,
            PRIMARY KEY (repo, issue_number, name)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kg_raw_relations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            repo            TEXT    NOT NULL,
            issue_number    INTEGER NOT NULL,
            source_name     TEXT    NOT NULL,
            target_name     TEXT    NOT NULL,
            type            TEXT    NOT NULL,
            description     TEXT,
            weight          REAL    DEFAULT 1.0,
            extracted_at    TEXT
        )
    """)
    conn.commit()


def fetch_unprocessed(conn: sqlite3.Connection, limit: int | None = None,
                      repo: str | None = None) -> list[dict]:
    """Return unprocessed issues as a list of dicts, optionally filtered by repo."""
    query = """
        SELECT repo, issue_number, title, body, labels, state,
               html_url, user_login, comments_json
        FROM issues
        WHERE processed_at IS NULL
    """
    params: list = []
    if repo:
        query += " AND repo LIKE ?"
        params.append(f"%{repo}%")
    query += " ORDER BY issue_number ASC"
    if limit:
        query += f" LIMIT {int(limit)}"

    cols = ["repo", "issue_number", "title", "body", "labels", "state",
            "html_url", "user_login", "comments_json"]
    rows = conn.execute(query, params).fetchall()
    return [dict(zip(cols, row)) for row in rows]


def save_analysis(conn: sqlite3.Connection, repo: str, issue_number: int,
                  result: dict) -> None:
    """Insert analysis result and mark issue as processed."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO analysis
            (repo, issue_number, relevance, relevance_reason, summary,
             root_cause, solution, lessons_json, tags_json, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        repo,
        issue_number,
        result.get("relevance", 0),
        result.get("relevance_reason"),
        result.get("summary"),
        result.get("root_cause"),
        result.get("solution"),
        json.dumps(result.get("lessons") or [], ensure_ascii=False),
        json.dumps(result.get("tags") or [], ensure_ascii=False),
        now,
    ))
    conn.execute(
        "UPDATE issues SET processed_at = ? WHERE repo = ? AND issue_number = ?",
        (now, repo, issue_number),
    )
    conn.commit()


def save_kg_data(conn: sqlite3.Connection, repo: str, issue_number: int,
                 result: dict) -> int:
    """Store per-issue KG entities and relations. Returns count of entities saved."""
    now = datetime.now(timezone.utc).isoformat()
    entities = result.get("entities") or []
    relations = result.get("relations") or []

    # Clear previous KG data for this issue (allows re-extraction)
    conn.execute(
        "DELETE FROM kg_raw_entities WHERE repo = ? AND issue_number = ?",
        (repo, issue_number),
    )
    conn.execute(
        "DELETE FROM kg_raw_relations WHERE repo = ? AND issue_number = ?",
        (repo, issue_number),
    )

    saved = 0
    entity_names = set()
    for ent in entities:
        name = (ent.get("name") or "").strip()
        etype = (ent.get("type") or "").strip()
        desc = (ent.get("description") or "").strip()
        if not name or etype not in VALID_ENTITY_TYPES:
            continue
        entity_names.add(name)
        conn.execute("""
            INSERT OR REPLACE INTO kg_raw_entities
                (repo, issue_number, name, type, description, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (repo, issue_number, name, etype, desc, now))
        saved += 1

    for rel in relations:
        src = (rel.get("source") or "").strip()
        tgt = (rel.get("target") or "").strip()
        rtype = (rel.get("type") or "").strip()
        desc = (rel.get("description") or "").strip()
        weight = rel.get("weight", 1.0)
        if not src or not tgt or rtype not in VALID_RELATION_TYPES:
            continue
        if isinstance(weight, (int, float)):
            weight = max(0.0, min(1.0, float(weight)))
        else:
            weight = 1.0
        conn.execute("""
            INSERT INTO kg_raw_relations
                (repo, issue_number, source_name, target_name, type,
                 description, weight, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (repo, issue_number, src, tgt, rtype, desc, weight, now))

    conn.commit()
    return saved


def has_kg_data(conn: sqlite3.Connection, repo: str, issue_number: int) -> bool:
    """Check if an issue already has KG entities extracted."""
    row = conn.execute(
        "SELECT COUNT(*) FROM kg_raw_entities WHERE repo = ? AND issue_number = ?",
        (repo, issue_number),
    ).fetchone()
    return row[0] > 0


def fetch_analyzed_without_kg(conn: sqlite3.Connection, limit: int | None = None,
                              repo: str | None = None,
                              min_relevance: int = 30) -> list[dict]:
    """Return analyzed issues that don't have KG data yet."""
    query = """
        SELECT a.repo, a.issue_number, i.title, a.summary, a.root_cause,
               a.solution, a.lessons_json, a.tags_json, a.relevance
        FROM analysis a
        JOIN issues i ON a.repo = i.repo AND a.issue_number = i.issue_number
        WHERE a.relevance >= ?
          AND NOT EXISTS (
              SELECT 1 FROM kg_raw_entities k
              WHERE k.repo = a.repo AND k.issue_number = a.issue_number
          )
    """
    params: list = [min_relevance]
    if repo:
        query += " AND a.repo LIKE ?"
        params.append(f"%{repo}%")
    query += " ORDER BY a.issue_number ASC"
    if limit:
        query += f" LIMIT {int(limit)}"

    cols = ["repo", "issue_number", "title", "summary", "root_cause",
            "solution", "lessons_json", "tags_json", "relevance"]
    rows = conn.execute(query, params).fetchall()
    return [dict(zip(cols, row)) for row in rows]


def build_kg_extract_text(analysis: dict) -> str:
    """Build a condensed text from existing analysis for KG-only extraction."""
    parts = [
        f"# {analysis['title']}",
        f"Repo: {analysis['repo']}  Issue: #{analysis['issue_number']}",
        "",
        f"## Summary\n{analysis.get('summary') or 'N/A'}",
        f"\n## Root Cause\n{analysis.get('root_cause') or 'Unknown'}",
        f"\n## Solution\n{analysis.get('solution') or 'Unknown'}",
    ]
    lessons = json.loads(analysis.get("lessons_json") or "[]")
    if lessons:
        parts.append("\n## Lessons Learned")
        for lesson in lessons:
            parts.append(f"- {lesson}")
    tags = json.loads(analysis.get("tags_json") or "[]")
    if tags:
        parts.append(f"\n## Tags: {', '.join(tags)}")
    return "\n".join(parts)


def extract_kg_only(client: OpenAI, text: str) -> dict:
    """Send condensed analysis to LLM for KG-only extraction (cheaper)."""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": KG_EXTRACT_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


# ── Thread builder ────────────────────────────────────────────

def build_thread_text(issue: dict) -> str:
    """Assemble the full issue thread into a single text block."""
    parts = [
        f"# {issue['title']}",
        f"State: {issue['state']}  |  Labels: {issue['labels'] or 'none'}",
        f"Author: {issue['user_login']}",
        "",
        issue["body"] or "(no description)",
    ]

    comments = json.loads(issue["comments_json"] or "[]")
    if comments:
        parts.append("\n--- Comments ---\n")
        for c in comments:
            parts.append(f"**{c.get('user', 'unknown')}** ({c.get('created_at', '')}):")
            parts.append(c.get("body", ""))
            parts.append("")

    return "\n".join(parts)


# ── LLM analysis ─────────────────────────────────────────────

def analyse_issue(client: OpenAI, thread_text: str) -> dict:
    """Send one issue to the LLM and return parsed JSON."""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": thread_text},
        ],
        temperature=0.2,
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


# ── Markdown output ───────────────────────────────────────────

def slug_from_repo(repo: str) -> str:
    """'espressif/esp-protocols' → 'esp-protocols'."""
    return repo.split("/")[-1]


def write_markdown(issue: dict, result: dict) -> Path:
    """Write a knowledge markdown file for a relevant issue."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    slug = slug_from_repo(issue["repo"])
    filename = f"{slug}-{issue['issue_number']}.md"
    path = KNOWLEDGE_DIR / filename

    lessons = result.get("lessons") or []
    tags = result.get("tags") or []

    relevance = result.get("relevance", 0)
    relevance_reason = result.get("relevance_reason", "")

    lines = [
        f"# {issue['title']}",
        "",
        f"**Issue:** [{issue['repo']}#{issue['issue_number']}]({issue['html_url']})  ",
        f"**State:** {issue['state']}  |  **Labels:** {issue['labels'] or 'none'}  ",
        f"**Relevance:** {relevance}% — {relevance_reason}",
        "",
        "## Summary",
        "",
        result.get("summary") or "N/A",
        "",
        "## Root Cause",
        "",
        result.get("root_cause") or "Unknown / not identified.",
        "",
        "## Solution",
        "",
        result.get("solution") or "Not resolved / unknown.",
        "",
        "## Lessons Learned",
        "",
    ]
    if lessons:
        for lesson in lessons:
            lines.append(f"- {lesson}")
    else:
        lines.append("- (none)")
    lines += [
        "",
        "## Tags",
        "",
        ", ".join(f"`{t}`" for t in tags) if tags else "(none)",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ── Main ──────────────────────────────────────────────────────

def run_extract_kg(client: OpenAI, conn: sqlite3.Connection,
                   limit: int | None, repo: str | None,
                   min_relevance: int) -> int:
    """Extract KG entities/relations from already-analyzed issues (no re-analysis)."""
    issues = fetch_analyzed_without_kg(conn, limit=limit, repo=repo,
                                       min_relevance=min_relevance)
    total = len(issues)
    repo_msg = f" (repo filter: {repo})" if repo else ""
    print(f"Found {total} analyzed issue(s) needing KG extraction{repo_msg}.",
          file=sys.stderr, flush=True)

    stats = {"processed": 0, "entities": 0, "skipped_error": 0}

    for i, analysis in enumerate(issues, 1):
        r = analysis["repo"]
        num = analysis["issue_number"]
        title = (analysis.get("title") or "")[:80]
        print(f"\n[{i}/{total}] #{num} {title}", file=sys.stderr, flush=True)

        try:
            text = build_kg_extract_text(analysis)
            kg_result = extract_kg_only(client, text)
        except json.JSONDecodeError as exc:
            print(f"  ✗ JSON parse error: {exc}", file=sys.stderr, flush=True)
            stats["skipped_error"] += 1
            continue
        except Exception as exc:
            print(f"  ✗ API error: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            stats["skipped_error"] += 1
            continue

        try:
            n_ent = save_kg_data(conn, r, num, kg_result)
            n_rel = len(kg_result.get("relations") or [])
            print(f"  entities={n_ent}  relations={n_rel}", file=sys.stderr, flush=True)
            stats["entities"] += n_ent
        except Exception as exc:
            print(f"  ✗ DB error: {exc}", file=sys.stderr, flush=True)
            stats["skipped_error"] += 1
            continue

        stats["processed"] += 1

    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(
        f"KG extraction done: processed {stats['processed']}, "
        f"total entities {stats['entities']}, "
        f"errors {stats['skipped_error']}",
        file=sys.stderr, flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse harvested esp-modem issues with LLM")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of issues to process (default: all)")
    parser.add_argument("--repo", type=str, default=None,
                        help="Filter by repo substring, e.g. 'esp-protocols' or 'esp-idf'")
    parser.add_argument("--min-relevance", type=int, default=60,
                        help="Min relevance score to write markdown (default: 60)")
    parser.add_argument("--db", type=str, default=DB_PATH,
                        help="Path to harvest.db")
    parser.add_argument("--extract-kg", action="store_true",
                        help="Extract KG entities/relations from already-analyzed issues "
                             "(uses condensed analysis as input — cheaper than full re-analysis)")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set in .env", file=sys.stderr, flush=True)
        return 1

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    conn = sqlite3.connect(args.db)
    init_analysis_table(conn)
    init_kg_tables(conn)

    # ── KG-only extraction mode ───────────────────────────────
    if args.extract_kg:
        rc = run_extract_kg(client, conn, args.limit, args.repo, args.min_relevance)
        conn.close()
        return rc

    # ── Normal analysis mode (now includes KG extraction) ─────
    issues = fetch_unprocessed(conn, limit=args.limit, repo=args.repo)
    total = len(issues)
    repo_msg = f" (repo filter: {args.repo})" if args.repo else ""
    print(f"Found {total} unprocessed issue(s) to analyse{repo_msg}.", file=sys.stderr, flush=True)

    stats = {"processed": 0, "kept": 0, "low_relevance": 0, "skipped_error": 0,
             "kg_entities": 0}

    for i, issue in enumerate(issues, 1):
        repo = issue["repo"]
        num = issue["issue_number"]
        title = issue["title"][:80]
        print(f"\n[{i}/{total}] #{num} {title}", file=sys.stderr, flush=True)

        try:
            thread_text = build_thread_text(issue)
            result = analyse_issue(client, thread_text)
        except json.JSONDecodeError as exc:
            print(f"  ✗ JSON parse error: {exc}", file=sys.stderr, flush=True)
            stats["skipped_error"] += 1
            continue
        except Exception as exc:
            print(f"  ✗ API error: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            stats["skipped_error"] += 1
            continue

        relevance = result.get("relevance", 0)
        reason = result.get("relevance_reason", "")
        tags = result.get("tags") or []
        print(f"  relevance={relevance}%  tags={tags}", file=sys.stderr, flush=True)
        if reason:
            print(f"  reason: {reason}", file=sys.stderr, flush=True)

        try:
            save_analysis(conn, repo, num, result)
            n_ent = save_kg_data(conn, repo, num, result)
            if n_ent:
                print(f"  KG: {n_ent} entities", file=sys.stderr, flush=True)
                stats["kg_entities"] += n_ent
        except Exception as exc:
            print(f"  ✗ DB error: {exc}", file=sys.stderr, flush=True)
            stats["skipped_error"] += 1
            continue

        if relevance >= args.min_relevance:
            md_path = write_markdown(issue, result)
            print(f"  → {md_path}", file=sys.stderr, flush=True)
            stats["kept"] += 1
        else:
            stats["low_relevance"] += 1

        stats["processed"] += 1

    # Summary
    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(
        f"Done: processed {stats['processed']}, "
        f"kept {stats['kept']} (>={args.min_relevance}%), "
        f"low relevance {stats['low_relevance']}, "
        f"errors {stats['skipped_error']}, "
        f"KG entities {stats['kg_entities']}",
        file=sys.stderr, flush=True,
    )

    total_analysed = conn.execute("SELECT COUNT(*) FROM analysis").fetchone()[0]
    total_kept = conn.execute(
        "SELECT COUNT(*) FROM analysis WHERE relevance >= ?",
        (args.min_relevance,),
    ).fetchone()[0]
    total_kg_ent = conn.execute("SELECT COUNT(*) FROM kg_raw_entities").fetchone()[0]
    total_kg_rel = conn.execute("SELECT COUNT(*) FROM kg_raw_relations").fetchone()[0]
    print(
        f"Total in analysis table: {total_analysed} ({total_kept} with relevance >= {args.min_relevance}%)",
        file=sys.stderr, flush=True,
    )
    print(
        f"Total KG: {total_kg_ent} raw entities, {total_kg_rel} raw relations",
        file=sys.stderr, flush=True,
    )

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
