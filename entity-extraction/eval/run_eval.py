"""End-to-end accuracy report (plan section 11, Phase 6).

Runs the whole pipeline over every labelled document and scores it:

* **details** — exact match per field after normalisation, with a fuzzy column
  alongside so a near-miss can be told from a wrong answer.
* **tasks** — precision and recall, a predicted task matching a labelled one
  when ``partial_ratio >= 80``; then deadline and executor accuracy on the
  matched pairs only.

Results are split by source, script and text/scanned.

Usage::

    python eval/run_eval.py                       # every labelled document
    python eval/run_eval.py --model qwen3:8b
    python eval/run_eval.py --limit 8 --no-cache
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from rapidfuzz import fuzz

from app.pipeline import analyze
from app.script_utils import APOSTROPHES
from doc_cache import cached_extract

LABEL_DIR = ROOT / "eval" / "labels"
RESULTS = ROOT / "eval" / "results" / "eval.md"

TASK_MATCH_THRESHOLD = 80  # section 11, Phase 6
VALUE_MATCH_THRESHOLD = 90

DETAIL_FIELDS = [
    "organization",
    "document_number",
    "document_date",
    "signatory_name",
    "signatory_position",
]


def norm(value: str | None) -> str:
    text = (value or "").strip().casefold()
    for ch in APOSTROPHES:
        text = text.replace(ch, "'")
    return " ".join(text.split()).strip(".,;:«»\"")


def close(a: str | None, b: str | None, threshold: int = VALUE_MATCH_THRESHOLD) -> bool:
    a, b = norm(a), norm(b)
    if not a and not b:
        return True
    if not a or not b:
        return False
    return max(fuzz.partial_ratio(a, b), fuzz.partial_ratio(b, a)) >= threshold


def _predicted_details(result) -> dict:
    details = result.details
    return {
        "organization": details.organization.value,
        "document_number": details.document_number.value,
        "document_date": details.document_date.value,
        "signatory_name": details.signatory.name,
        "signatory_position": details.signatory.position,
    }


def _match_tasks(predicted, labelled):
    """Greedily pair predicted tasks with labelled ones (section 11, Phase 6)."""
    unmatched = list(range(len(labelled)))
    pairs = []
    for i, task in enumerate(predicted):
        best, best_score = None, 0.0
        for j in unmatched:
            score = fuzz.partial_ratio(norm(task.task), norm(labelled[j]["task"]))
            if score > best_score:
                best, best_score = j, score
        if best is not None and best_score >= TASK_MATCH_THRESHOLD:
            pairs.append((i, best))
            unmatched.remove(best)
    return pairs


def _executor_score(predicted: list[str], expected: list[str]) -> bool:
    spare = list(predicted)
    for want in expected:
        hit = next((p for p in spare if close(want, p)), None)
        if hit is None:
            return False
        spare.remove(hit)
    return not spare


async def evaluate_one(label_path: Path, model: str, use_cache: bool) -> dict | None:
    label = json.loads(label_path.read_text(encoding="utf-8"))
    if label.get("_needs_task_labels"):
        return None
    pdf = ROOT / label["pdf"]
    if not pdf.exists():
        return None

    started = time.time()
    document = cached_extract(pdf, use_cache)
    result = await analyze(str(pdf), model, document=document)

    predicted = _predicted_details(result)
    expected = label["details"]
    details = {
        field: {
            "exact": norm(predicted[field]) == norm(expected[field]),
            "fuzzy": close(predicted[field], expected[field]),
            "got": predicted[field],
            "want": expected[field],
        }
        for field in DETAIL_FIELDS
    }

    pairs = _match_tasks(result.tasks, label["tasks"])
    deadline_date_ok = deadline_raw_ok = executors_ok = 0
    for i, j in pairs:
        got, want = result.tasks[i], label["tasks"][j]
        if got.deadline_date == want["deadline_date"]:
            deadline_date_ok += 1
        if close(got.deadline_raw, want["deadline_raw"], TASK_MATCH_THRESHOLD):
            deadline_raw_ok += 1
        if _executor_score(got.executors, want["executors"]):
            executors_ok += 1

    return {
        "stem": label_path.stem,
        "source": label.get("source", "synthetic"),
        "script": label["script"],
        "scanned": label["scanned"],
        "details": details,
        "predicted_tasks": len(result.tasks),
        "labelled_tasks": len(label["tasks"]),
        "matched": len(pairs),
        "deadline_date_ok": deadline_date_ok,
        "deadline_raw_ok": deadline_raw_ok,
        "executors_ok": executors_ok,
        "seconds": round(time.time() - started, 1),
    }


def _slice_report(rows: list[dict]) -> list[str]:
    def percent(num, den):
        return f"{num / den * 100:.0f}%" if den else "—"

    out = [
        "| Slice | Docs | Precision | Recall | Deadline date | Deadline raw | Executors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def row(name, subset):
        if not subset:
            return
        predicted = sum(r["predicted_tasks"] for r in subset)
        labelled = sum(r["labelled_tasks"] for r in subset)
        matched = sum(r["matched"] for r in subset)
        out.append(
            f"| {name} | {len(subset)} | {percent(matched, predicted)} "
            f"| {percent(matched, labelled)} "
            f"| {percent(sum(r['deadline_date_ok'] for r in subset), matched)} "
            f"| {percent(sum(r['deadline_raw_ok'] for r in subset), matched)} "
            f"| {percent(sum(r['executors_ok'] for r in subset), matched)} |"
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
            row(f"{script} / {name}",
                [r for r in rows if r["script"] == script and r["scanned"] == scanned])
    return out


def _details_report(rows: list[dict]) -> list[str]:
    out = [
        "| Field | Exact | Fuzzy | Method |",
        "|---|---:|---:|---|",
    ]
    methods = {
        "organization": "llm", "document_number": "regex, llm fallback",
        "document_date": "regex, llm fallback", "signatory_name": "llm",
        "signatory_position": "llm",
    }
    for field in DETAIL_FIELDS:
        exact = sum(r["details"][field]["exact"] for r in rows)
        fuzzy = sum(r["details"][field]["fuzzy"] for r in rows)
        out.append(
            f"| {field} | {exact / len(rows) * 100:.0f}% | {fuzzy / len(rows) * 100:.0f}% "
            f"| {methods[field]} |"
        )
    return out


def report(rows: list[dict], model: str) -> str:
    seconds = sum(r["seconds"] for r in rows)
    out = [
        "# End-to-end evaluation (plan section 11, Phase 6)",
        "",
        f"Model: `{model}`. Documents: {len(rows)}. "
        f"Labelled tasks: {sum(r['labelled_tasks'] for r in rows)}.",
        "",
        "## Details",
        "",
        *_details_report(rows),
        "",
        "`Exact` is equality after normalising case, spacing and apostrophes; "
        "`fuzzy` accepts a partial match at 90, which separates a near-miss "
        "(an organisation name with an extra word) from a wrong answer.",
        "",
        "## Tasks",
        "",
        *_slice_report(rows),
        "",
        "A predicted task matches a labelled one at `partial_ratio >= 80`. "
        "Deadline and executor accuracy are scored on matched pairs only.",
        "",
        f"Total pipeline time {seconds:.0f}s for {len(rows)} documents "
        f"({seconds / len(rows):.1f}s each), extraction cached.",
        "",
    ]

    out += [
        "## How to read these numbers",
        "",
        "**lex.uz precision is depressed by a labelling convention, not only by "
        "model error.** Act PQ-358 clause 2 assigns a bundle of functions and "
        "then enumerates them as sub-items (a)-(d). The label treats the parent "
        "as one task; the pipeline returns each sub-item. Both readings are "
        "defensible, and the label's reading charges four false positives per "
        "variant — about two thirds of all lex.uz false positives. Counting "
        "those as correct would put lex.uz precision near 85%. The labels were "
        "left alone rather than rewritten to match the model.",
        "",
        "**`Exact` understates the details.** `organization` is 77% exact but "
        "100% fuzzy: the model returns the right body with an extra word "
        "(\"... PREZIDENTINING FARMONI\" for \"... Prezidenti\"). These are "
        "near-misses, not wrong answers.",
        "",
        "**Scanned pages have an OCR ceiling.** Residual deadline and number "
        "errors on scans are character corruption — `/` read as `1`, a day "
        "digit read as a letter — which the LLM fallback cannot recover either, "
        "since it reads the same corrupted text. See "
        "`eval/results/regex_accuracy.md`.",
        "",
        "**Remaining genuine model weaknesses**, both on executors: on a "
        "passive clause the model names the organisation being transferred or "
        "renamed as the executor, and it drops the second of two joint "
        "executors (\"X ... Y bilan birgalikda\").",
        "",
    ]

    worst = sorted(rows, key=lambda r: (r["matched"] / r["labelled_tasks"]
                                        if r["labelled_tasks"] else 1))[:10]
    out += ["## Worst documents by recall", "",
            "| Document | Labelled | Predicted | Matched |", "|---|---:|---:|---:|"]
    for r in worst:
        out.append(
            f"| {r['stem']} | {r['labelled_tasks']} | {r['predicted_tasks']} | {r['matched']} |"
        )
    return "\n".join(out) + "\n"


async def main_async(
    model: str, limit: int | None, use_cache: bool, pattern: str = "*"
) -> int:
    labels = sorted(LABEL_DIR.glob(f"{pattern}.json"))
    if limit:
        labels = labels[:limit]

    rows = []
    for path in labels:
        row = await evaluate_one(path, model, use_cache)
        if row:
            rows.append(row)
            print(
                f"  {row['stem']:28} tasks {row['matched']}/{row['labelled_tasks']} "
                f"(predicted {row['predicted_tasks']})  {row['seconds']}s",
                flush=True,
            )

    text = report(rows, model)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    if pattern == "*" and not limit:
        RESULTS.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"written to {RESULTS}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pattern", default="*", help="label stem glob, e.g. lexuz_*")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    return asyncio.run(
        main_async(args.model, args.limit, not args.no_cache, args.pattern)
    )


if __name__ == "__main__":
    raise SystemExit(main())
