"""Evaluate one or more systems and write reports/eval_results.{md,json}.

Usage: python -m scripts.evaluate [--systems random,...] [--primary NAME]
       python -m scripts.evaluate --sweep [--grid '{"rrf_k": [5, 60]}']
       python -m scripts.evaluate --tune-reranker [--grid '{"rrf": [1], "fuzzy": [0, 1]}']
       python -m scripts.evaluate --calibrate   (thresholds → reports/thresholds.{md,json})
The primary system (default: the last one) gets the worst-failure list and the predictions log.
--sweep, --tune-reranker and --calibrate use decision sets only (validation + realistic-dev), never
test numbers.
"""

import argparse
import itertools
import json
import logging

import numpy as np

from app.core.config import get_settings
from app.eval.calibration import calibrate, collect_tables, threshold_table
from app.eval.report import write_reports
from app.eval.runner import build_context, evaluate, sweep
from app.eval.systems import SYSTEMS
from app.eval.tuning import tune_reranker

DEFAULT_GRID = {
    "answer_index_weight": [0.0, 0.1, 0.25],
    "rrf_k": [5, 20, 60],
    "tfidf_entity_weight": [0.25, 0.5, 1.0],
    "use_bm25": [True, False],
}


def run_sweep(grid: dict[str, list]) -> None:
    settings = get_settings()
    ctx = build_context(settings)
    keys = list(grid)
    combos = [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]
    combos = [c | {"use_answer_index": c.get("answer_index_weight", 1) > 0} for c in combos]
    rows = sweep(ctx, combos)
    rows.sort(
        key=lambda r: (
            -r["realistic_dev"].get("top1", 0),
            -r["realistic_dev"].get("mrr10", 0),
            -r["validation"]["top1"],
        )
    )
    print(f"{'dev Top-1':>9} {'dev MRR':>8} {'val Top-1':>9} {'val MRR':>8}  overrides")
    for r in rows:
        dev, val = r["realistic_dev"], r["validation"]
        print(
            f"{dev.get('top1', 0):9.3f} {dev.get('mrr10', 0):8.3f} {val['top1']:9.3f} "
            f"{val['mrr10']:8.3f}  {r['overrides']}"
        )
    out = settings.reports_dir / "sweep.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


RERANK_GRID = {
    "rrf": [0.5, 1.0, 2.0],
    "sim_q": [0.0, 0.5, 1.0, 2.0],
    "sim_a": [0.0, 0.5],
    "sim_e": [0.0, 0.5, 1.0, 2.0],
    "fuzzy": [0.0, 0.5, 1.0, 2.0],
    "entity_match": [0.0, 0.5, 1.0, 2.0],
    "entity_match_q": [0.0, 0.5, 1.0, 2.0],
    "category": [0.0, 0.5],
    "entity_conflict": [0.0, 1.0, 3.0],
}


def run_rerank_tuning(grid: dict[str, list[float]]) -> None:
    settings = get_settings()
    ctx = build_context(settings)
    baseline, rows = tune_reranker(ctx, grid)
    b_dev, b_val = baseline["realistic_dev"], baseline["validation"]
    print(
        f"baseline (fused only): dev Top-1 {b_dev['top1']:.3f} MRR {b_dev['mrr10']:.3f} | "
        f"val Top-1 {b_val['top1']:.3f}"
    )
    print(
        f"{len(rows)} combinations kept\n{'dev Top-1':>9} {'dev MRR':>8} {'val Top-1':>9}  weights"
    )
    for r in rows[:15]:
        weights = {k: v for k, v in r["weights"].items() if v}
        print(
            f"{r['realistic_dev']['top1']:9.3f} {r['realistic_dev']['mrr10']:8.3f} "
            f"{r['validation']['top1']:9.3f}  {weights}"
        )
    out = settings.reports_dir / "rerank_sweep.json"
    out.write_text(
        json.dumps({"baseline": baseline, "top": rows[:100]}, indent=2), encoding="utf-8"
    )
    print(f"\nSaved {out}")


def run_calibration() -> None:
    settings = get_settings()
    ctx = build_context(settings)
    dev, val, ood = collect_tables(ctx, settings.ood_dev_path)
    grid = np.round(np.arange(0.30, 0.96, 0.01), 2)
    # OOD not_found floor 0.80 = the user's "balanced" choice (Phase 7): suggestions for an
    # off-topic question are cheap, a third of realistic queries rejected (floor 0.90) is not.
    result = calibrate(
        dev, val, ood, grid, margin_grid=[0.0, 0.02, 0.05, 0.1], min_ood_not_found=0.80
    )
    if result is None:
        print("No thresholds meet the constraints; relax min_precision / max_ood_answer.")
        return

    def pct(v: float | None) -> str:
        return "–" if v is None else f"{100 * v:.1f}"

    head = [
        "answer ≥",
        "dev coverage",
        "dev precision",
        "dev useful",
        "val coverage",
        "val precision",
        "OOD answered",
        "OOD not_found",
    ]
    rows = threshold_table(dev, val, ood, result.min_margin, result.suggest_threshold)
    lines = [
        "# Threshold calibration",
        "",
        "Decision sets only: validation (template paraphrases), realistic-dev (hand-written) and",
        "OOD-dev (`data/ood_dev.txt`). Rule: highest dev coverage with precision when answering",
        "≥ 95% on dev and validation and ≤ 5% of OOD-dev answered; then the suggest threshold with",
        'the highest dev "useful" rate (correct answer, or gold in the top-3 suggestions) such',
        "that ≥ 80% of OOD-dev gets not_found (a suggestion for an off-topic question is cheap;",
        "a 90% floor would reject a third of realistic queries — see the table below).",
        "",
        f"**Chosen:** `ANSWER_THRESHOLD={result.answer_threshold:.2f}`, "
        f"`SUGGEST_THRESHOLD={result.suggest_threshold:.2f}`, "
        f"`ANSWER_MIN_MARGIN={result.min_margin:.2f}`",
        "",
        "| set | coverage | precision when answering | suggest | not_found | useful |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in (
        ("realistic-dev", result.dev),
        ("validation", result.validation),
        ("OOD-dev", result.ood),
    ):
        lines.append(
            f"| {name} | {pct(m['coverage'])} | {pct(m['precision_at_answer'])} | "
            f"{pct(m['suggest_rate'])} | {pct(m['not_found_rate'])} | {pct(m['useful'])} |"
        )
    lines += [
        "",
        f"Trade-off at margin {result.min_margin:.2f} (suggest threshold capped at the "
        "answer threshold), percentages:",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    lines += ["| " + " | ".join([f"{r[0]:.2f}", *[pct(v) for v in r[1:]]]) + " |" for r in rows]
    out = settings.reports_dir / "thresholds.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (settings.reports_dir / "thresholds.json").write_text(
        json.dumps(result.__dict__, indent=2), encoding="utf-8"
    )
    print("\n".join(lines[8:16]))
    print(f"\nReport: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default=",".join(SYSTEMS), help="comma-separated names")
    parser.add_argument("--primary", default=None, help="system for the failure analysis")
    parser.add_argument("--sweep", action="store_true", help="grid search on decision sets")
    parser.add_argument("--tune-reranker", action="store_true", help="re-ranker weight search")
    parser.add_argument("--calibrate", action="store_true", help="decision threshold calibration")
    parser.add_argument("--grid", default=None, help="JSON dict of setting/feature -> values")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("bm25s").setLevel(logging.WARNING)

    if args.calibrate:
        run_calibration()
        return
    if args.tune_reranker:
        run_rerank_tuning(json.loads(args.grid) if args.grid else RERANK_GRID)
        return
    if args.sweep:
        run_sweep(json.loads(args.grid) if args.grid else DEFAULT_GRID)
        return

    settings = get_settings()
    names = [n.strip() for n in args.systems.split(",") if n.strip()]
    ctx, reports = evaluate(settings, names)
    primary = next((r for r in reports if r.name == args.primary), reports[-1])
    results = write_reports(ctx, reports, primary, settings.reports_dir)

    for name, m in results["systems"].items():
        val, test = m["validation"], m["test"]
        print(
            f"{name:>20}  val Top-1 {val['top1']:.3f}  test Top-1 {test['top1']:.3f}  "
            f"Top-3 {test['top3']:.3f}  MRR@10 {test['mrr10']:.3f}  "
            f"p95 {test['latency_p95_ms']:.2f} ms"
        )
    print(f"\nReport: {settings.reports_dir / 'eval_results.md'}")


if __name__ == "__main__":
    main()
