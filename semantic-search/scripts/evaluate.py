"""Evaluate one or more systems and write reports/eval_results.{md,json}.

Usage: python -m scripts.evaluate [--systems random,...] [--primary NAME]
       python -m scripts.evaluate --sweep [--grid '{"rrf_k": [5, 60]}']
The primary system (default: the last one) gets the worst-failure list and the predictions log.
--sweep prints decision metrics only (validation + realistic-dev), never test numbers.
"""

import argparse
import itertools
import json
import logging

from app.core.config import get_settings
from app.eval.report import write_reports
from app.eval.runner import build_context, evaluate, sweep
from app.eval.systems import SYSTEMS

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default=",".join(SYSTEMS), help="comma-separated names")
    parser.add_argument("--primary", default=None, help="system for the failure analysis")
    parser.add_argument("--sweep", action="store_true", help="grid search on decision sets")
    parser.add_argument("--grid", default=None, help="JSON dict of setting -> values")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("bm25s").setLevel(logging.WARNING)

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
