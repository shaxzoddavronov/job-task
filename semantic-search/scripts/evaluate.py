"""Evaluate one or more systems and write reports/eval_results.{md,json}.

Usage: python -m scripts.evaluate [--systems random,...] [--primary NAME]
The primary system (default: the last one) gets the worst-failure list and the predictions log.
"""

import argparse
import logging

from app.core.config import get_settings
from app.eval.report import write_reports
from app.eval.runner import evaluate
from app.eval.systems import SYSTEMS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default=",".join(SYSTEMS), help="comma-separated names")
    parser.add_argument("--primary", default=None, help="system for the failure analysis")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

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
