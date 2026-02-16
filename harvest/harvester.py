#!/usr/bin/env python3
"""
Stage 1: Harvest esp-modem related issues from GitHub.

Scans espressif/esp-idf and espressif/esp-protocols for issues
mentioning modem-related keywords. Stores matching issues in a
local SQLite database with incremental update support.
"""
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from github import Github

# ── Config ────────────────────────────────────────────────────

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
DB_PATH = os.path.join(os.path.dirname(__file__), "harvest.db")

REPOS = [
    "espressif/esp-idf",
    "espressif/esp-protocols",
]

KEYWORDS = [
    r"\bPPP\b",
    r"\bPPPoS\b",
    r"\besp[-_]modem\b",
    r"\bAT\s*command",
    r"\bDCE\b",
    r"\bDTE\b",
    r"\bmodem\b",
    r"\bCMUX\b",
    r"\bppp_netif\b",
    r"\besp_netif_ppp\b",
]

# Pre-compile a single combined pattern (case-insensitive)
_KW_PATTERN = re.compile("|".join(KEYWORDS), re.IGNORECASE)


# ── Database ──────────────────────────────────────────────────

def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create the issues table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS issues (
            repo            TEXT    NOT NULL,
            issue_number    INTEGER NOT NULL,
            title           TEXT,
            body            TEXT,
            labels          TEXT,
            state           TEXT,
            html_url        TEXT,
            user_login      TEXT,
            comments_json   TEXT,
            created_at      TEXT,
            updated_at      TEXT,
            fetched_at      TEXT,
            processed_at    TEXT,
            PRIMARY KEY (repo, issue_number)
        )
    """)
    conn.commit()
    return conn


def get_stored_updated_at(conn: sqlite3.Connection, repo: str, issue_number: int) -> str | None:
    """Return the stored updated_at timestamp for an issue, or None."""
    row = conn.execute(
        "SELECT updated_at FROM issues WHERE repo = ? AND issue_number = ?",
        (repo, issue_number),
    ).fetchone()
    return row[0] if row else None


def upsert_issue(conn: sqlite3.Connection, data: dict) -> None:
    """Insert or replace an issue row."""
    conn.execute("""
        INSERT OR REPLACE INTO issues
            (repo, issue_number, title, body, labels, state, html_url,
             user_login, comments_json, created_at, updated_at, fetched_at, processed_at)
        VALUES
            (:repo, :issue_number, :title, :body, :labels, :state, :html_url,
             :user_login, :comments_json, :created_at, :updated_at, :fetched_at, NULL)
    """, data)


# ── Keyword matching ──────────────────────────────────────────

def matches_keywords_basic(issue) -> bool:
    """Quick check on title + body + labels (no extra API call)."""
    texts = [
        issue.title or "",
        issue.body or "",
    ]
    for label in issue.labels:
        texts.append(label.name)
    blob = "\n".join(texts)
    return _KW_PATTERN.search(blob) is not None


def matches_keywords_with_comments(comments: list[dict]) -> bool:
    """Check comment bodies for keywords."""
    blob = "\n".join(c.get("body", "") for c in comments)
    return _KW_PATTERN.search(blob) is not None


# ── Harvesting ────────────────────────────────────────────────

def fetch_comments(issue) -> list[dict]:
    """Fetch all comments for an issue as plain dicts."""
    return [
        {
            "user": c.user.login if c.user else "unknown",
            "body": c.body or "",
            "created_at": c.created_at.isoformat(),
        }
        for c in issue.get_comments()
    ]


def harvest_repo(g: Github, repo_name: str, conn: sqlite3.Connection) -> dict:
    """
    Walk every issue in a repo. Store those that match keywords.
    Returns stats dict.
    """
    repo = g.get_repo(repo_name)
    stats = {"scanned": 0, "matched": 0, "skipped_unchanged": 0}

    # Fetch all issues (open + closed), newest first
    issues = repo.get_issues(state="all", sort="updated", direction="desc")

    for issue in issues:
        # Skip pull requests (GitHub API returns PRs as issues)
        if issue.pull_request:
            continue

        stats["scanned"] += 1
        num = issue.number
        gh_updated = issue.updated_at.isoformat() if issue.updated_at else ""

        # Incremental: skip if we already have this version
        stored = get_stored_updated_at(conn, repo_name, num)
        if stored == gh_updated:
            stats["skipped_unchanged"] += 1
            if stats["scanned"] % 500 == 0:
                print(f"  ... scanned {stats['scanned']} issues ({stats['matched']} matched)", file=sys.stderr)
            continue

        # Two-pass keyword filter:
        # 1) Quick check on title + body + labels (free — already fetched)
        basic_match = matches_keywords_basic(issue)

        # 2) Only fetch comments if basic match, or for small repos always check
        if basic_match:
            comments = fetch_comments(issue)
        else:
            # Skip expensive comment fetch for non-matching issues
            if stats["scanned"] % 500 == 0:
                print(f"  ... scanned {stats['scanned']} issues ({stats['matched']} matched)", file=sys.stderr)
            continue

        stats["matched"] += 1

        labels_str = ",".join(l.name for l in issue.labels)
        data = {
            "repo": repo_name,
            "issue_number": num,
            "title": issue.title,
            "body": issue.body or "",
            "labels": labels_str,
            "state": issue.state,
            "html_url": issue.html_url,
            "user_login": issue.user.login if issue.user else "unknown",
            "comments_json": json.dumps(comments, ensure_ascii=False),
            "created_at": issue.created_at.isoformat() if issue.created_at else "",
            "updated_at": gh_updated,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        upsert_issue(conn, data)
        conn.commit()

        print(
            f"  #{num} [{issue.state}] {issue.title[:80]}  "
            f"({len(comments)} comments)",
            file=sys.stderr,
        )

    return stats


def main() -> int:
    if not GITHUB_TOKEN:
        print("Error: GITHUB_TOKEN not set in .env", file=sys.stderr)
        return 1

    g = Github(GITHUB_TOKEN, per_page=100)
    conn = init_db()

    for repo_name in REPOS:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Harvesting: {repo_name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        stats = harvest_repo(g, repo_name, conn)

        print(
            f"\nDone: scanned {stats['scanned']}, "
            f"matched {stats['matched']}, "
            f"skipped (unchanged) {stats['skipped_unchanged']}",
            file=sys.stderr,
        )

    # Summary
    total = conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
    print(f"\nTotal issues in database: {total}", file=sys.stderr)

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
