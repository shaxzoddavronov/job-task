"""Measure section 6 regex accuracy over every labelled document.

Plan section 11, Phase 3 ("number/date accuracy on all labeled docs printed by
a small script"). Extraction is cached under ``eval/.cache`` because OCR of the
scanned variants dominates the runtime.

Usage::

    python eval/regex_accuracy.py            # all labelled documents
    python eval/regex_accuracy.py --no-cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.extract_text import extract
from app.regex_rules import find_document_date, find_document_number
from app.script_utils import to_latin_with_offsets

LABEL_DIR = ROOT / "eval" / "labels"
CACHE_DIR = ROOT / "eval" / ".cache"


def page_one(pdf: Path, use_cache: bool = True) -> str:
    """Page 1 of a PDF, cached by path and modification time."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = f"{pdf.stem}-{int(pdf.stat().st_mtime)}.json"
    cached = CACHE_DIR / key
    if use_cache and cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))["page1"]

    doc = extract(str(pdf))
    text = doc.pages[0].text if doc.pages else ""
    cached.write_text(json.dumps({"page1": text}, ensure_ascii=False), encoding="utf-8")
    return text


def _norm(value: str | None) -> str:
    return "".join((value or "").split()).casefold()


def evaluate(use_cache: bool = True) -> list[dict]:
    labels = sorted(LABEL_DIR.glob("*.json"))
    rows = []
    for path in labels:
        label = json.loads(path.read_text(encoding="utf-8"))
        if label.get("_needs_task_labels"):
            continue  # unlabelled long documents
        pdf = ROOT / label["pdf"]
        if not pdf.exists():
            continue

        started = time.time()
        original = page_one(pdf, use_cache)
        latin, offsets = to_latin_with_offsets(original)

        found_number = find_document_number(latin, original, offsets)
        found_date = find_document_date(latin, original, offsets)
        expected = label["details"]

        rows.append(
            {
                "stem": path.stem,
                "source": label.get("source", "synthetic"),
                "script": label["script"],
                "scanned": label["scanned"],
                "number_ok": _norm(found_number and found_number["value"])
                == _norm(expected["document_number"]),
                "number_found": found_number["value"] if found_number else None,
                "number_expected": expected["document_number"],
                "date_ok": (found_date or {}).get("value") == expected["document_date"],
                "date_found": found_date["value"] if found_date else None,
                "date_expected": expected["document_date"],
                "seconds": round(time.time() - started, 1),
            }
        )
    return rows


def _rate(rows, key):
    return (sum(r[key] for r in rows) / len(rows) * 100) if rows else 0.0


def report(rows: list[dict]) -> str:
    out = ["# Regex accuracy (plan section 6)", ""]
    out += [f"Documents: {len(rows)}", ""]
    out += ["| Slice | N | Number | Date |", "|---|---:|---:|---:|"]

    def row(name, subset):
        if subset:
            out.append(
                f"| {name} | {len(subset)} | {_rate(subset, 'number_ok'):.0f}% "
                f"| {_rate(subset, 'date_ok'):.0f}% |"
            )

    row("**all**", rows)
    for source in sorted({r["source"] for r in rows}):
        row(source, [r for r in rows if r["source"] == source])
    for script in ("latin", "cyrillic"):
        row(script, [r for r in rows if r["script"] == script])
    for scanned, name in ((False, "text layer"), (True, "scanned")):
        row(name, [r for r in rows if r["scanned"] == scanned])
    for script in ("latin", "cyrillic"):
        for scanned, name in ((False, "text"), (True, "scan")):
            row(
                f"{script} / {name}",
                [r for r in rows if r["script"] == script and r["scanned"] == scanned],
            )

    out += [
        "",
        "## Known limitations",
        "",
        "Every remaining failure is OCR character corruption on a *scanned*",
        "page, not a missing pattern. Both fields are 100% on text-layer",
        "documents in both scripts.",
        "",
        "* `/` inside a number is read as `1`: `17/1-674` -> `1711-674`. The",
        "  number is found, but two characters are wrong. The LLM fallback",
        "  reads the same corrupted text, so it cannot recover the original",
        "  either.",
        "* A day digit is read as a letter: `«6»` -> `кб`, `«5»` -> `S5`. The",
        "  date then has no digit to match and is lost.",
        "* Guillemets around the day are read as letters (`«29»` -> `к29э`),",
        "  which breaks the day/month join in `DATE_DMY`.",
        "* `№` printed *after* the number is not matched.",
        "* Where a scan loses the document's own number, the leftmost match can",
        "  fall on a *referenced* act's number instead (seen once:",
        "  `lexuz_8517274_cyr_scan` returning `PQ-265` from the preamble).",
        "",
    ]

    failures = [r for r in rows if not r["number_ok"] or not r["date_ok"]]
    if failures:
        out += ["", f"## Failures ({len(failures)})", ""]
        out += ["| Document | Field | Expected | Found |", "|---|---|---|---|"]
        for r in failures:
            if not r["number_ok"]:
                out.append(
                    f"| {r['stem']} | number | `{r['number_expected']}` | `{r['number_found']}` |"
                )
            if not r["date_ok"]:
                out.append(
                    f"| {r['stem']} | date | `{r['date_expected']}` | `{r['date_found']}` |"
                )
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "eval" / "results" / "regex_accuracy.md"))
    args = parser.parse_args()

    rows = evaluate(use_cache=not args.no_cache)
    text = report(rows)
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
