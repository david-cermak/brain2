#!/usr/bin/env python3
"""
Fetch a GitHub issue or pull request thread and save as markdown (issue_N.md / pr_N.md).
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from github import Auth, Github

load_dotenv()


def _format_thread(title: str, author_login: str, body: str | None, comments) -> str:
    parts: list[str] = []
    parts.append(f"# {title}\n")
    parts.append(f"@{author_login}:")
    parts.append(body or "")
    parts.append("\n---\n")
    for comment in comments:
        parts.append(f"@{comment.user.login}:")
        parts.append(comment.body or "")
        parts.append("\n---\n")
    return "\n".join(parts)


def _format_review_comments(review_comments) -> str:
    """Group inline code review comments by file and format as markdown."""
    from collections import defaultdict

    by_file: dict[str, list] = defaultdict(list)
    for rc in review_comments:
        by_file[rc.path].append(rc)

    if not by_file:
        return ""

    parts: list[str] = ["\n## Code Review Comments\n"]
    for filepath in sorted(by_file):
        parts.append(f"### `{filepath}`\n")
        comments_in_file = sorted(by_file[filepath], key=lambda c: (c.original_line or 0, c.created_at))
        for rc in comments_in_file:
            line_info = f" (line {rc.original_line})" if rc.original_line else ""
            parts.append(f"**@{rc.user.login}**{line_info}:")
            if rc.diff_hunk:
                parts.append(f"```diff\n{rc.diff_hunk}\n```")
            parts.append(rc.body or "")
            parts.append("\n---\n")
    return "\n".join(parts)


def _format_reviews(reviews) -> str:
    """Format PR reviews (approval / change-request / review-level comments)."""
    review_list = [r for r in reviews if r.body or r.state != "COMMENTED"]
    if not review_list:
        return ""

    parts: list[str] = ["\n## Reviews\n"]
    for review in review_list:
        state_label = review.state.replace("_", " ").title()
        parts.append(f"**@{review.user.login}** ({state_label}):")
        if review.body:
            parts.append(review.body)
        parts.append("\n---\n")
    return "\n".join(parts)


def export_issue(repo, number: int) -> str:
    issue = repo.get_issue(number=number)
    return _format_thread(
        issue.title,
        issue.user.login,
        issue.body,
        issue.get_comments(),
    )


def export_pr(repo, number: int) -> str:
    pr = repo.get_pull(number)
    thread = _format_thread(
        pr.title,
        pr.user.login,
        pr.body,
        pr.get_issue_comments(),
    )
    reviews = _format_reviews(pr.get_reviews())
    code_comments = _format_review_comments(pr.get_review_comments())
    return thread + reviews + code_comments


def main() -> int:
    default_repo = os.getenv("GITHUB_REPO", "espressif/esp-protocols")
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Export a GitHub issue or PR conversation to a markdown file.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--issue",
        type=int,
        metavar="N",
        help="Issue number to export",
    )
    group.add_argument(
        "--pr",
        type=int,
        metavar="N",
        help="Pull request number to export",
    )
    parser.add_argument(
        "--repo",
        default=default_repo,
        help=f"owner/repo (default: from GITHUB_REPO or {default_repo!r})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help=f"Directory for issue_N.md / pr_N.md (default: {script_dir})",
    )

    args = parser.parse_args()
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        warnings.warn(
            "GITHUB_TOKEN is not set; using unauthenticated API access (lower rate limits, may be slower).",
            stacklevel=2,
        )

    g = Github(auth=Auth.Token(token)) if token else Github()
    repo = g.get_repo(args.repo)

    if args.issue is not None:
        n = args.issue
        prefix = "issue"
        text = export_issue(repo, n)
    else:
        n = args.pr
        prefix = "pr"
        text = export_pr(repo, n)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{prefix}_{n}.md"
    out_path.write_text(text, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
