"""Render evaluation results to reports/eval_results.{md,json} and a predictions log."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from app.eval.metrics import QueryRecord
from app.eval.runner import EvalContext, SystemReport

WORST_N = 20


def _fmt(value, pct: bool = True) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{100 * value:.1f}" if pct else f"{value:.1f}"
    return str(value)


def _table(headers: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return (
        lines
        + ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows]
        + [""]
    )


def _ranking_row(name: str, m: dict, ent: dict | None = None) -> list:
    row = [name, m.get("n", 0), _fmt(m.get("top1")), _fmt(m.get("top3")), _fmt(m.get("mrr10"))]
    if ent is not None:
        row.append(_fmt(ent.get("top1")))
    return row + [
        _fmt(m.get("coverage")),
        _fmt(m.get("precision_at_answer")),
        _fmt(m.get("latency_p50_ms"), pct=False),
        _fmt(m.get("latency_p95_ms"), pct=False),
    ]


def _preview(ctx: EvalContext, answer_id: int | None, n: int = 80) -> str:
    if answer_id is None or answer_id not in ctx.corpus.answers:
        return "–"
    text = ctx.corpus.answers[answer_id].text
    return text if len(text) <= n else text[: n - 1] + "…"


def worst_failures(records: list[QueryRecord], n: int = WORST_N) -> list[QueryRecord]:
    """Wrong top-1 predictions, most confident (and answered) first."""
    wrong = [r for r in records if r.top1 != r.gold]
    return sorted(wrong, key=lambda r: (r.status != "answer", -r.confidence))[:n]


def render_markdown(ctx: EvalContext, reports: list[SystemReport], primary: SystemReport) -> str:
    s = ctx.split
    md = [
        "# Evaluation results",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC · seed {ctx.settings.eval_seed} · "
        f"{len(ctx.corpus.answers)} answers · index {len(s.index)} / validation "
        f"{len(s.validation)} / test {len(s.test)} questions · "
        f"thresholds answer ≥ {ctx.settings.answer_threshold}, suggest ≥ "
        f"{ctx.settings.suggest_threshold}, margin ≥ {ctx.settings.answer_min_margin}",
        "",
        "> The random split is **optimistic**: most paraphrases are template variants of one",
        "> original question, so a held-out question usually has a near-copy in the index.",
        "> The realistic set (hand-written queries) is the better estimate of real use.",
        "",
        "Percentages except latency (ms). Coverage = share answered; Prec@ans = top-1 accuracy",
        "among answered queries. Tune on validation, report test.",
        "",
        "## Validation (for decisions)",
        "",
    ]
    head = ["system", "n", "Top-1", "Top-3", "MRR@10", "Coverage", "Prec@ans", "p50", "p95"]
    md += _table(head, [_ranking_row(r.name, r.metrics["validation"]) for r in reports])

    md += ["## Test (for reporting)", ""]
    head_t = head[:5] + ["Entity Top-1"] + head[5:]
    md += _table(
        head_t,
        [_ranking_row(r.name, r.metrics["test"], r.metrics["test_entity"]) for r in reports],
    )

    md += ["## Safety checks (test-derived)", ""]
    rows = []
    for r in reports:
        m = r.metrics
        rows.append(
            [
                r.name,
                f"{_fmt(m['ood'].get('not_found_rate'))} (n={m['ood']['n']})",
                f"{_fmt(m['law_injection'].get('not_found_rate'))} (n={m['law_injection']['n']})",
                f"{_fmt(m['law_injection'].get('kept_original_answer_rate'))}",
                f"{_fmt(m['law_swap'].get('kept_original_answer_rate'))} (n={m['law_swap']['n']})",
                f"{_fmt(m['org_swap'].get('kept_original_answer_rate'))} (n={m['org_swap']['n']})",
            ]
        )
    md += _table(
        [
            "system",
            "OOD → not_found ↑",
            "Unknown law → not_found ↑",
            "Unknown law: kept answer ↓",
            "Law swap: kept answer ↓",
            "Org swap: kept answer (diagnostic)",
        ],
        rows,
    )

    if ctx.realistic:
        md += ["## Realistic set (hand-written queries, full index)", ""]
        md += _table(head, [_ranking_row(r.name, r.metrics["realistic"]) for r in reports])
        kinds = list(primary.metrics.get("realistic_by_kind", {}))
        if kinds:
            md += ["Top-1 by query kind:", ""]
            md += _table(
                [
                    "system",
                    *[f"{k} (n={primary.metrics['realistic_by_kind'][k]['n']})" for k in kinds],
                ],
                [
                    [r.name, *[_fmt(r.metrics["realistic_by_kind"][k]["top1"]) for k in kinds]]
                    for r in reports
                ],
            )

    md += [f"## {WORST_N} worst test failures — `{primary.name}`", ""]
    rows = []
    for r in worst_failures(primary.records["test"]):
        rows.append(
            [
                r.query,
                f"{r.gold}: {_preview(ctx, r.gold)}",
                f"{r.top1}: {_preview(ctx, r.top1)}",
                f"{r.confidence:.3f}",
                r.status,
                r.rank or "–",
            ]
        )
    md += _table(["query", "expected", "predicted", "score", "status", "gold rank"], rows)
    return "\n".join(md)


def write_reports(
    ctx: EvalContext, reports: list[SystemReport], primary: SystemReport, out_dir: Path
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    s = ctx.split
    results = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": ctx.settings.eval_seed,
        "dataset": {
            "answers": len(ctx.corpus.answers),
            "questions": len(ctx.corpus.questions),
            "index": len(s.index),
            "validation": len(s.validation),
            "test": len(s.test),
            "realistic": len(ctx.realistic),
        },
        "thresholds": {
            "answer": ctx.settings.answer_threshold,
            "suggest": ctx.settings.suggest_threshold,
            "min_margin": ctx.settings.answer_min_margin,
        },
        "primary": primary.name,
        "systems": {
            r.name: {"build_seconds": round(r.build_seconds, 3), **r.metrics} for r in reports
        },
    }
    (out_dir / "eval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "eval_results.md").write_text(
        render_markdown(ctx, reports, primary), encoding="utf-8"
    )
    with (out_dir / "eval_predictions.jsonl").open("w", encoding="utf-8") as f:
        for set_name, records in primary.records.items():
            for r in records:
                f.write(json.dumps({"set": set_name, **asdict(r)}, ensure_ascii=False) + "\n")
    return results
