"""Threshold calibration for the answer / suggest / not_found decision (decision sets only).

The engine runs once per query (validation, realistic-dev, OOD-dev); the decision is then
replayed offline for every (answer_threshold, suggest_threshold, min_margin) on the grid,
using the same rule as engine.decide_status.

Selection
1. answer_threshold + min_margin: highest realistic-dev coverage such that precision when
   answering is >= min_precision on realistic-dev and validation, and at most max_ood_answer
   of OOD-dev queries are answered.
2. suggest_threshold (<= answer_threshold): highest realistic-dev "useful" rate (answered
   correctly, or suggested with the gold answer in the top 3) such that at least
   min_ood_not_found of OOD-dev queries get not_found.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.engine import QAEngine
from app.eval.metrics import QueryRecord
from app.eval.runner import EvalContext, load_ood, run_queries


@dataclass
class ScoreTable:
    s1: np.ndarray  # top-1 score
    s2: np.ndarray  # top-2 score (0 if absent)
    correct: np.ndarray  # top-1 is the gold answer (False for OOD)
    hit3: np.ndarray  # gold in the top 3 (False for OOD)

    @classmethod
    def from_records(cls, records: list[QueryRecord]) -> "ScoreTable":
        def score(r: QueryRecord, i: int) -> float:
            return r.ranked[i][1] if len(r.ranked) > i else 0.0

        return cls(
            s1=np.array([score(r, 0) for r in records]),
            s2=np.array([score(r, 1) for r in records]),
            correct=np.array([r.gold is not None and r.top1 == r.gold for r in records]),
            hit3=np.array([r.rank is not None and r.rank <= 3 for r in records]),
        )


def decide(t: ScoreTable, answer: float, suggest: float, margin: float) -> tuple[np.ndarray, ...]:
    """Masks (answered, suggested, not_found), same rule as engine.decide_status."""
    not_found = t.s1 < suggest
    answered = ~not_found & (t.s1 >= answer) & ((t.s1 - t.s2) >= margin)
    return answered, ~not_found & ~answered, not_found


def decision_metrics(t: ScoreTable, answer: float, suggest: float, margin: float) -> dict:
    answered, suggested, not_found = decide(t, answer, suggest, margin)
    n = max(len(t.s1), 1)
    return {
        "coverage": float(answered.sum() / n),
        "precision_at_answer": float(t.correct[answered].mean()) if answered.any() else None,
        "suggest_rate": float(suggested.sum() / n),
        "not_found_rate": float(not_found.sum() / n),
        "useful": float(((answered & t.correct) | (suggested & t.hit3)).sum() / n),
    }


@dataclass
class Calibration:
    answer_threshold: float
    suggest_threshold: float
    min_margin: float
    dev: dict
    validation: dict
    ood: dict


def calibrate(
    dev: ScoreTable,
    val: ScoreTable,
    ood: ScoreTable,
    answer_grid: np.ndarray,
    margin_grid: list[float],
    min_precision: float = 0.95,
    max_ood_answer: float = 0.05,
    min_ood_not_found: float = 0.90,
) -> Calibration | None:
    best = None
    for a in answer_grid:
        for m in margin_grid:
            d, v = decision_metrics(dev, a, 0.0, m), decision_metrics(val, a, 0.0, m)
            o = decision_metrics(ood, a, 0.0, m)
            ok = (
                d["precision_at_answer"] is not None
                and d["precision_at_answer"] >= min_precision
                and (v["precision_at_answer"] or 0) >= min_precision
                and o["coverage"] <= max_ood_answer
            )
            key = (d["coverage"], d["precision_at_answer"] or 0, -a)
            if ok and (best is None or key > best[0]):
                best = (key, float(a), float(m))
    if best is None:
        return None
    _, a, m = best

    best_s = None
    for s in answer_grid[answer_grid <= a]:
        d, o = decision_metrics(dev, a, s, m), decision_metrics(ood, a, s, m)
        key = (d["useful"], s)  # ties → the more conservative (higher) threshold
        if o["not_found_rate"] >= min_ood_not_found and (best_s is None or key > best_s[0]):
            best_s = (key, float(s))
    s = best_s[1] if best_s else a
    return Calibration(
        a,
        s,
        m,
        decision_metrics(dev, a, s, m),
        decision_metrics(val, a, s, m),
        decision_metrics(ood, a, s, m),
    )


def collect_tables(ctx: EvalContext, ood_dev_path: Path) -> tuple[ScoreTable, ...]:
    k = ctx.settings.eval_top_k
    val = run_queries(
        QAEngine(ctx.index_corpus, ctx.settings), ctx.query_sets["validation"], ctx.extractor, k
    )
    full = QAEngine(ctx.corpus, ctx.settings)
    dev = run_queries(full, ctx.realistic_dev, ctx.extractor, k)
    ood_queries = load_ood(ctx.settings.model_copy(update={"ood_questions_path": ood_dev_path}))
    ood = run_queries(full, ood_queries, ctx.extractor, k)
    return tuple(ScoreTable.from_records(r) for r in (dev, val, ood))


def threshold_table(
    dev: ScoreTable, val: ScoreTable, ood: ScoreTable, margin: float, suggest: float
) -> list[list]:
    """Precision / coverage trade-off for a range of answer thresholds (fixed margin)."""
    rows = []
    for a in np.round(np.arange(0.40, 0.96, 0.05), 2):
        d = decision_metrics(dev, a, min(suggest, a), margin)
        v = decision_metrics(val, a, min(suggest, a), margin)
        o = decision_metrics(ood, a, min(suggest, a), margin)
        rows.append(
            [
                a,
                d["coverage"],
                d["precision_at_answer"],
                d["useful"],
                v["coverage"],
                v["precision_at_answer"],
                o["coverage"],
                o["not_found_rate"],
            ]
        )
    return rows
