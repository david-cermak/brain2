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

Analyse the following GitHub issue thread and return a single JSON object \
with exactly these keys:

{
  "relevant": <bool — true if the issue is actually about esp-modem, PPP, \
modem AT commands, CMUX, DTE/DCE, or cellular connectivity on ESP32; \
false if it is unrelated>,
  "summary": "<one paragraph describing the problem>",
  "root_cause": "<what went wrong — null if unknown>",
  "solution": "<how it was resolved — null if unresolved or unknown>",
  "lessons": ["<actionable takeaway 1>", "..."],
  "tags": ["<topic tag>", "..."]
}

Rules:
- Return ONLY valid JSON, no markdown fences, no commentary.
- "tags" should be short lowercase labels like "ppp", "cmux", "usb modem", \
"carrier loss", "at commands", "dce/dte", "netif", "ota", "memory leak", etc.
- "lessons" should be concrete and useful for a developer building with esp-modem.
- If the issue is not relevant, still fill summary but set root_cause, \
solution, and lessons to null/empty.
"""


# ── Database ──────────────────────────────────────────────────

def init_analysis_table(conn: sqlite3.Connection) -> None:
    """Create the analysis table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis (
            repo            TEXT    NOT NULL,
            issue_number    INTEGER NOT NULL,
            relevant        BOOLEAN,
            summary         TEXT,
            root_cause      TEXT,
            solution        TEXT,
            lessons_json    TEXT,
            tags_json       TEXT,
            analyzed_at     TEXT,
            PRIMARY KEY (repo, issue_number)
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
            (repo, issue_number, relevant, summary, root_cause, solution,
             lessons_json, tags_json, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        repo,
        issue_number,
        result.get("relevant", False),
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

    lines = [
        f"# {issue['title']}",
        "",
        f"**Issue:** [{issue['repo']}#{issue['issue_number']}]({issue['html_url']})  ",
        f"**State:** {issue['state']}  |  **Labels:** {issue['labels'] or 'none'}",
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

def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse harvested esp-modem issues with LLM")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of issues to process (default: all)")
    parser.add_argument("--repo", type=str, default=None,
                        help="Filter by repo substring, e.g. 'esp-protocols' or 'esp-idf'")
    parser.add_argument("--db", type=str, default=DB_PATH,
                        help="Path to harvest.db")
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

    issues = fetch_unprocessed(conn, limit=args.limit, repo=args.repo)
    total = len(issues)
    repo_msg = f" (repo filter: {args.repo})" if args.repo else ""
    print(f"Found {total} unprocessed issue(s) to analyse{repo_msg}.", file=sys.stderr, flush=True)

    stats = {"processed": 0, "relevant": 0, "skipped_error": 0}

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

        relevant = result.get("relevant", False)
        tags = result.get("tags") or []
        print(f"  relevant={relevant}  tags={tags}", file=sys.stderr, flush=True)

        try:
            save_analysis(conn, repo, num, result)
        except Exception as exc:
            print(f"  ✗ DB error: {exc}", file=sys.stderr, flush=True)
            stats["skipped_error"] += 1
            continue

        if relevant:
            md_path = write_markdown(issue, result)
            print(f"  → {md_path}", file=sys.stderr, flush=True)
            stats["relevant"] += 1

        stats["processed"] += 1

    # Summary
    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(
        f"Done: processed {stats['processed']}, "
        f"relevant {stats['relevant']}, "
        f"errors {stats['skipped_error']}",
        file=sys.stderr, flush=True,
    )

    total_analysed = conn.execute("SELECT COUNT(*) FROM analysis").fetchone()[0]
    total_relevant = conn.execute("SELECT COUNT(*) FROM analysis WHERE relevant = 1").fetchone()[0]
    print(f"Total in analysis table: {total_analysed} ({total_relevant} relevant)", file=sys.stderr, flush=True)

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
