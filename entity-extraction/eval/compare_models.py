"""Compare candidate models on labelled clauses (plan section 11, Phase 4).

Runs the Call B prompt over a stratified sample of clauses whose ground truth
is known, and reports per-model accuracy for ``is_task``, ``deadline`` and
``executors``, plus latency and throughput.

The sample is stratified by script and by whether the clause is a task, so a
model cannot score well by answering "task" to everything.

Usage::

    python eval/compare_models.py --sample 120
    python eval/compare_models.py --models qwen3:4b qwen3:8b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx

from app.llm import Client
from app.script_utils import OKINA, TURNED_COMMA

LABEL_DIR = ROOT / "eval" / "labels"
RESULTS = ROOT / "eval" / "results" / "model_comparison.md"

CANDIDATES = [
    "qwen3:4b",
    "qwen3:8b",
    "qwen3:30b-a3b",
    "hf.co/mradermacher/uzbek-llama-3.1-8B-instruct-v2-GGUF:Q4_K_M",
    "hf.co/kmamaroziqov/alloma-8b-q4:Q4_K_M",
]


# --- the labelled clause set ----------------------------------------------


def load_clauses() -> list[dict]:
    """Clause-level items with known answers.

    Synthetic documents contribute both tasks and non-tasks; the labelled
    lex.uz documents contribute real task clauses. Only text-layer variants are
    used — this measures the *model*, not OCR.
    """
    items = []
    for path in sorted(LABEL_DIR.glob("doc_*.json")):
        label = json.loads(path.read_text(encoding="utf-8"))
        if label["scanned"]:
            continue
        context = ""
        for clause in label["clauses"]:
            items.append(
                {
                    "text": clause["text"],
                    "context": context,
                    "script": label["script"],
                    "source": "synthetic",
                    "is_task": clause["is_task"],
                    "deadline": clause["deadline_raw"],
                    "executors": clause["executors"],
                }
            )
            # A parent clause ending in ":" supplies context to *all* of its
            # sub-items, until the next top-level numbered clause closes the
            # group. Resetting after the first sub-item (as this did
            # originally) starved (b), (v), ... of the executor named in the
            # parent, which is exactly the case section 7.3 exists for.
            text = clause["text"].strip()
            if text.endswith(":"):
                context = text
            elif re.match(r"^\d+[.)]", text):
                context = ""

    for path in sorted(LABEL_DIR.glob("lexuz_*.json")):
        label = json.loads(path.read_text(encoding="utf-8"))
        if label["scanned"] or label.get("_needs_task_labels"):
            continue
        for task in label["tasks"]:
            items.append(
                {
                    "text": task["task"],
                    "context": "",
                    "script": label["script"],
                    "source": "lexuz",
                    "is_task": True,
                    "deadline": task["deadline_raw"],
                    "executors": task["executors"],
                }
            )
    return items


def stratified(items: list[dict], size: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[tuple, list[dict]] = {}
    for item in items:
        buckets.setdefault((item["script"], item["is_task"]), []).append(item)

    per_bucket = max(1, size // len(buckets))
    sample = []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        sample.extend(bucket[:per_bucket])
    rng.shuffle(sample)
    return sample


# --- scoring ---------------------------------------------------------------


def norm(value: str | None) -> str:
    """Compare on content, not on apostrophe style or spacing."""
    text = (value or "").strip().casefold()
    for ch in ("'", "`", "‘", "’", OKINA, TURNED_COMMA):
        text = text.replace(ch, "'")
    return " ".join(text.split()).rstrip(".,;:")


#: Executors are matched the way section 7.5 grounds them — fuzzily. Exact set
#: equality would reject "Bosh vaziri A.N. Aripov" against "Oʻzbekiston
#: Respublikasi Bosh vaziri A.N. Aripov", which is the same person.
EXECUTOR_THRESHOLD = 90


def _fuzzy_match(want: list[str], have: list[str]) -> tuple[int, list[str]]:
    """Greedily pair expected executors with predicted ones."""
    from rapidfuzz import fuzz

    spare = list(have)
    matched = 0
    for expected in want:
        for i, candidate in enumerate(spare):
            close = max(
                fuzz.partial_ratio(expected, candidate),
                fuzz.partial_ratio(candidate, expected),
            )
            if close >= EXECUTOR_THRESHOLD:
                matched += 1
                spare.pop(i)
                break
    return matched, spare


def score(expected: dict, got: dict | None) -> dict:
    if got is None:
        return {"is_task_ok": False, "deadline_ok": False, "executors_ok": False,
                "executor_overlap": 0.0, "failed": True}

    is_task_ok = bool(got.get("is_task")) == expected["is_task"]
    result = {"is_task_ok": is_task_ok, "failed": False,
              "deadline_ok": None, "executors_ok": None, "executor_overlap": None}

    if not expected["is_task"]:
        return result  # nothing else to grade on a non-task

    result["deadline_ok"] = norm(got.get("deadline")) == norm(expected["deadline"])

    want = [norm(e) for e in expected["executors"]]
    have = [norm(e) for e in (got.get("executors") or [])]
    matched, spare = _fuzzy_match(want, have)
    result["executors_ok"] = matched == len(want) and not spare
    denominator = len(want) + len(spare)
    result["executor_overlap"] = (matched / denominator) if denominator else 1.0
    return result


# --- running ---------------------------------------------------------------


async def run_model(model: str, sample: list[dict], concurrency: int) -> dict:
    client = Client(model=model, concurrency=concurrency)
    limit = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as http:
        try:  # warm up so the first clause is not charged for the model load
            await client.clause(http, "1. hisobot taqdim etilsin.")
        except Exception as exc:
            return {"model": model, "error": str(exc)[:120]}
        client.usage = type(client.usage)()

        async def one(item):
            async with limit:
                try:
                    return await client.clause(http, item["text"], item["context"])
                except Exception:
                    return None

        started = time.time()
        answers = await asyncio.gather(*(one(i) for i in sample))
        elapsed = time.time() - started

    scores = [score(item, answer) for item, answer in zip(sample, answers)]
    tasks = [s for s, i in zip(scores, sample) if i["is_task"]]

    def rate(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) * 100 if values else 0.0

    return {
        "model": model,
        "n": len(sample),
        "is_task": rate([s["is_task_ok"] for s in scores]),
        "deadline": rate([s["deadline_ok"] for s in tasks]),
        "executors": rate([s["executors_ok"] for s in tasks]),
        "executor_overlap": rate([s["executor_overlap"] for s in tasks]),
        "failed": sum(s["failed"] for s in scores),
        "seconds": elapsed,
        "per_clause": elapsed / len(sample),
        "tokens_per_second": client.usage.tokens_per_second,
        "prompt_tokens": client.usage.prompt_eval_count,
        "output_tokens": client.usage.eval_count,
    }


def report(rows: list[dict], sample: list[dict]) -> str:
    out = ["# Model comparison (plan section 11, Phase 4)", ""]
    out += [
        f"Clauses: {len(sample)} "
        f"({sum(i['is_task'] for i in sample)} tasks, "
        f"{sum(not i['is_task'] for i in sample)} non-tasks; "
        f"{sum(i['script'] == 'cyrillic' for i in sample)} Cyrillic). "
        "Text-layer documents only, so this measures the model and not OCR.",
        "",
        "| Model | is_task | deadline | executors | exec. overlap | s/clause | tok/s | failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: -(r.get("is_task") or 0)):
        if "error" in r:
            out.append(f"| `{r['model']}` | — | — | — | — | — | — | {r['error']} |")
            continue
        out.append(
            f"| `{r['model']}` | {r['is_task']:.0f}% | {r['deadline']:.0f}% | "
            f"{r['executors']:.0f}% | {r['executor_overlap']:.0f}% | "
            f"{r['per_clause']:.2f} | {r['tokens_per_second']:.0f} | {r['failed']} |"
        )
    out += [
        "",
        "`deadline` and `executors` are exact matches after normalising "
        "apostrophes, case and spacing, scored only on clauses that really are "
        "tasks. `exec. overlap` is the Jaccard overlap, which gives partial "
        "credit when one executor of several is missed.",
        "",
    ]
    return "\n".join(out)


async def main_async(models: list[str], size: int, concurrency: int) -> int:
    items = load_clauses()
    sample = stratified(items, size)
    print(f"{len(items)} labelled clauses available; evaluating {len(sample)}")

    rows = []
    for model in models:
        print(f"  {model} ...", flush=True)
        row = await run_model(model, sample, concurrency)
        rows.append(row)
        if "error" in row:
            print(f"    FAILED: {row['error']}")
        else:
            print(
                f"    is_task {row['is_task']:.0f}%  deadline {row['deadline']:.0f}%  "
                f"executors {row['executors']:.0f}%  {row['per_clause']:.2f}s/clause"
            )

    text = report(rows, sample)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"written to {RESULTS}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=CANDIDATES)
    parser.add_argument("--sample", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    return asyncio.run(main_async(args.models, args.sample, args.concurrency))


if __name__ == "__main__":
    raise SystemExit(main())
