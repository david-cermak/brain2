# Modem AT Command References

Per-modem AT command manuals, preprocessed from vendor PDFs into Markdown so
the triage agent (and other tooling) can grep / chunk / feed them to an LLM.

## Why preprocess at all

PDFs are unusable directly: the triage agent can't query them, and passing the
binary to an LLM wastes tokens and loses structure (tables, headings, command
syntax blocks). Markdown keeps:

- Section hierarchy (command index by chapter/command)
- Parameter tables (`<cid>`, `<PDP_type>`, etc. with full descriptions)
- Test/Read/Write/Execution command variants
- Example sessions

## Adding a new modem family

Whenever `esp-modem` gains support for a new modem (or a new manual revision
comes out), preprocess the vendor PDF once and commit the resulting Markdown.

```bash
# From repo root, with the knowledge venv active
python knowledge/preprocess_pdf.py path/to/vendor_manual.pdf

# Quick dry-run (first 20 pages) to verify table/heading quality
python knowledge/preprocess_pdf.py path/to/vendor_manual.pdf --pages 1-20 --dry-run

# Override the auto-derived slug
python knowledge/preprocess_pdf.py path/to/vendor_manual.pdf --slug quectel-eg915
```

Output lands in `knowledge/modems/<slug>.md` with a YAML frontmatter header
recording the source PDF name, SHA-256, page range, and generator version —
so re-runs are deterministic and diffs are meaningful.

## Tooling

[`preprocess_pdf.py`](../preprocess_pdf.py) wraps
[pymupdf4llm](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) with a few
defaults that work well for vendor AT command manuals:

| Option | Rationale |
|---|---|
| `use_ocr=False` | Vendor manuals are real PDFs; OCR triples the runtime for no quality gain. |
| `header=False, footer=False` | Drops repeating page chrome ("www.simcom.com", "N / 512", doc title banner). |
| Image placeholders stripped | `==> picture intentionally omitted <==` lines add noise; we don't need the decorative glyphs. |

The script is intentionally dumb: it doesn't call LLMs, doesn't chunk, doesn't
embed. Downstream agents do that on top of the generated Markdown.

## Contents

| File | Source PDF | Notes |
|---|---|---|
| [`sim7500-sim7600-v3.md`](sim7500-sim7600-v3.md) | `005073_SIM7500_SIM7600_Series_AT_Command_Manual_V3.00.pdf` | 512 pages, 1479 sections, covers SIMCom SIM7500/7600 series |

## Future: feeding the triage agent

Once triage (see [`../../triage/PROPOSAL.md`](../../triage/PROPOSAL.md)) is
wired up, these files become the ground-truth reference for:

- Validating that an AT command mentioned in an issue exists for the modem in
  question, and looking up its parameter semantics
- Expanding the KG with `modem_model → SUPPORTS → at_command` relations
  (deterministic, parsed from the `## N.M.P AT+XXX ...` heading pattern)
- Grounding LLM answers with exact command syntax / response formats rather
  than relying on memorized (and often wrong) AT command knowledge
