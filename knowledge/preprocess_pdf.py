"""Convert a modem AT command manual PDF to Markdown.

Usage:
    python preprocess_pdf.py <pdf_path> [--slug NAME] [--pages A-B]
                             [--keep-images] [--no-clean] [--dry-run]

Defaults to writing `knowledge/modems/<slug>.md`. The slug is derived from the
PDF filename if not given (e.g. `005073_SIM7500_SIM7600_Series_AT_Command_Manual_V3.00.pdf`
-> `sim7500-sim7600`). Use `--pages 1-10` for quick tests.

Why this exists:
    The triage agent will eventually consume per-modem AT command references.
    PDFs are unusable directly, so we preprocess them once (manually, per modem
    family) into clean Markdown that the agent can grep / chunk / feed to an LLM.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import re
import sys
from pathlib import Path

import pymupdf4llm

HERE = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = HERE / "modems"

IMAGE_PLACEHOLDER_RE = re.compile(
    r"^\s*\*\*==>\s*picture\s*\[[^\]]*\]\s*intentionally omitted\s*<==\*\*\s*$",
    re.MULTILINE,
)
MULTI_BLANK_RE = re.compile(r"\n{3,}")


def derive_slug(pdf_path: Path) -> str:
    """Turn a messy PDF filename into a short, stable slug.

    Heuristic: take the stem, lowercase, drop any leading numeric prefix
    (typical vendor document IDs), collapse non-alnum to `-`, then try to
    extract a `<vendorOrSeries>` token — we pick the longest group of
    alnum-with-underscore chunks that look like a part number (e.g. SIM7500).
    Falls back to the cleaned stem.
    """
    stem = pdf_path.stem.lower()
    stem = re.sub(r"^\d+[_\-]+", "", stem)
    parts = re.findall(r"[a-z0-9]+", stem)
    part_nos = [p for p in parts if re.fullmatch(r"[a-z]+[0-9]+[a-z0-9]*", p)]
    if part_nos:
        uniq: list[str] = []
        for p in part_nos:
            if p not in uniq:
                uniq.append(p)
        return "-".join(uniq)
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or "manual"


def parse_page_range(spec: str | None, total_pages: int) -> list[int] | None:
    if not spec:
        return None
    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", spec)
    if not m:
        raise ValueError(f"Invalid --pages value: {spec!r} (expected e.g. '1-10')")
    start = max(1, int(m.group(1)))
    end = min(total_pages, int(m.group(2)))
    if end < start:
        raise ValueError(f"--pages range is empty: {spec!r}")
    return list(range(start - 1, end))


def clean_markdown(md: str, *, strip_images: bool) -> str:
    if strip_images:
        md = IMAGE_PLACEHOLDER_RE.sub("", md)
    md = MULTI_BLANK_RE.sub("\n\n", md)
    return md.strip() + "\n"


def build_frontmatter(pdf_path: Path, *, slug: str, pages: list[int] | None,
                      total_pages: int, sha256: str) -> str:
    page_spec = (
        f"{pages[0] + 1}-{pages[-1] + 1}"
        if pages else f"1-{total_pages}"
    )
    return (
        "---\n"
        f"slug: {slug}\n"
        f"source_pdf: {pdf_path.name}\n"
        f"source_sha256: {sha256}\n"
        f"pages: {page_spec}\n"
        f"total_pages: {total_pages}\n"
        f"generated_at: {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}\n"
        f"generator: pymupdf4llm {pymupdf4llm.__version__}\n"
        "---\n\n"
    )


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def convert(pdf_path: Path, out_path: Path, *, pages_spec: str | None,
            keep_images: bool, clean: bool, dry_run: bool) -> Path:
    import pymupdf

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    with pymupdf.open(pdf_path) as doc:
        total_pages = doc.page_count
    pages = parse_page_range(pages_spec, total_pages)

    print(f"[preprocess] {pdf_path.name}: {total_pages} pages; "
          f"converting {'all' if pages is None else f'{len(pages)} pages'} ...",
          file=sys.stderr)

    md = pymupdf4llm.to_markdown(
        str(pdf_path),
        pages=pages,
        use_ocr=False,
        header=False,
        footer=False,
        show_progress=True,
    )

    if clean:
        md = clean_markdown(md, strip_images=not keep_images)

    frontmatter = build_frontmatter(
        pdf_path,
        slug=out_path.stem,
        pages=pages,
        total_pages=total_pages,
        sha256=sha256_file(pdf_path),
    )

    output = frontmatter + md

    if dry_run:
        print(output[:2000])
        print(f"\n[preprocess] dry-run; would write {len(output)} bytes to {out_path}",
              file=sys.stderr)
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    print(f"[preprocess] wrote {out_path} ({len(output):,} bytes)", file=sys.stderr)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("pdf", type=Path, help="Path to the AT command manual PDF")
    p.add_argument("--slug", help="Output slug (default: derived from filename)")
    p.add_argument("--output", type=Path,
                   help="Explicit output path (overrides --slug/default dir)")
    p.add_argument("--pages", help="Page range, e.g. '1-20' (1-indexed, inclusive)")
    p.add_argument("--keep-images", action="store_true",
                   help="Keep '==> picture intentionally omitted <==' placeholders")
    p.add_argument("--no-clean", action="store_true",
                   help="Skip post-processing (collapse blanks, strip placeholders)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print a preview instead of writing the file")
    args = p.parse_args(argv)

    slug = args.slug or derive_slug(args.pdf)
    out_path = args.output or (DEFAULT_OUT_DIR / f"{slug}.md")

    convert(
        args.pdf,
        out_path,
        pages_spec=args.pages,
        keep_images=args.keep_images,
        clean=not args.no_clean,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
