"""Grid search for the re-ranker weights on the decision sets (validation + realistic-dev).

Retrieval runs once per query; every weight combination is then scored in one vectorized pass
over a padded (queries × candidates × features) tensor, so thousands of combinations take
seconds. Test sets are never touched.
"""

import itertools
from dataclasses import dataclass

import numpy as np

from app.core.engine import QAEngine
from app.core.ranker import FEATURES, score_matrix
from app.eval.runner import EvalContext, EvalQuery


@dataclass
class FeatureBatch:
    features: np.ndarray  # (queries, max_candidates, n_features); padding rows are zeros
    valid: np.ndarray  # (queries, max_candidates) bool
    gold_pos: np.ndarray  # (queries,) index of the gold answer among candidates, -1 if absent
    has_entities: np.ndarray  # (queries,) bool


def collect_features(engine: QAEngine, queries: list[EvalQuery]) -> FeatureBatch:
    retrievals = [engine.retrieve(q.text) for q in queries]
    width = max((len(r.candidates) for r in retrievals), default=1) or 1
    features = np.zeros((len(queries), width, len(FEATURES)))
    valid = np.zeros((len(queries), width), dtype=bool)
    gold_pos = np.full(len(queries), -1)
    for i, (q, r) in enumerate(zip(queries, retrievals, strict=True)):
        for j, c in enumerate(r.candidates):
            features[i, j] = c.vector()
            valid[i, j] = True
            if c.answer_id == q.gold:
                gold_pos[i] = j
    has_entities = np.array([bool(r.tagged.tokens) for r in retrievals])
    return FeatureBatch(features, valid, gold_pos, has_entities)


def ranking_metrics(batch: FeatureBatch, weights: np.ndarray) -> dict[str, float]:
    """Top-1 and MRR@10 under `weights`; ties keep the fused order, like QAEngine.ask."""
    scores = score_matrix(batch.features, weights, batch.has_entities)
    scores = np.where(batch.valid, scores, -np.inf)
    n = len(batch.gold_pos)
    found = batch.gold_pos >= 0
    gold_idx = np.clip(batch.gold_pos, 0, None)
    gold_score = scores[np.arange(n), gold_idx][:, None]
    positions = np.arange(scores.shape[1])[None, :]
    ahead = (scores > gold_score) | ((scores == gold_score) & (positions < gold_idx[:, None]))
    rank = ahead.sum(axis=1) + 1
    rank = np.where(found, rank, np.iinfo(np.int64).max)
    return {
        "top1": float(np.mean(rank == 1)),
        "mrr10": float(np.mean(np.where(rank <= 10, 1.0 / rank, 0.0))),
    }


def weight_grid(grid: dict[str, list[float]]) -> list[np.ndarray]:
    """Cartesian product over the named features; unnamed features get weight 0."""
    unknown = set(grid) - set(FEATURES)
    if unknown:
        raise ValueError(f"Unknown features {sorted(unknown)}; available: {FEATURES}")
    names = list(grid)
    combos = []
    for values in itertools.product(*grid.values()):
        w = np.zeros(len(FEATURES))
        for name, value in zip(names, values, strict=True):
            w[FEATURES.index(name)] = value
        if w[: FEATURES.index("entity_conflict")].sum() > 0:
            combos.append(w)
    return combos


def tune_reranker(
    ctx: EvalContext, grid: dict[str, list[float]], max_val_drop: float = 0.005
) -> tuple[dict, list[dict]]:
    """Rank weight combinations by dev Top-1, dev MRR, validation Top-1.

    Combinations whose validation Top-1 falls more than `max_val_drop` below the fused-only
    baseline (re-ranker off) are dropped. Returns (baseline, ranked rows).
    """
    settings = ctx.settings
    val_engine = QAEngine(ctx.index_corpus, settings)
    val = collect_features(val_engine, ctx.query_sets["validation"])
    dev = collect_features(QAEngine(ctx.corpus, settings), ctx.realistic_dev)

    # Baseline: the model score alone (cross-encoder in rerank mode, bi-encoder in dense mode)
    model_only = np.zeros(len(FEATURES))
    model_only[FEATURES.index("rerank" if val_engine.mode == "rerank" else "dense")] = 1.0
    baseline = {
        "mode": val_engine.mode,
        "validation": ranking_metrics(val, model_only),
        "realistic_dev": ranking_metrics(dev, model_only),
    }

    rows = []
    for w in weight_grid(grid):
        row = {
            "weights": {name: float(v) for name, v in zip(FEATURES, w, strict=True)},
            "validation": ranking_metrics(val, w),
            "realistic_dev": ranking_metrics(dev, w),
        }
        if row["validation"]["top1"] >= baseline["validation"]["top1"] - max_val_drop:
            rows.append(row)
    rows.sort(
        key=lambda r: (
            -r["realistic_dev"]["top1"],
            -r["realistic_dev"]["mrr10"],
            -r["validation"]["top1"],
        )
    )
    return baseline, rows
