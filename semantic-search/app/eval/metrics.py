"""Ranking and decision metrics over evaluated queries."""

from dataclasses import dataclass, field

import numpy as np

from app.core.engine import Status


@dataclass
class QueryRecord:
    query: str
    gold: int | None  # None for out-of-domain queries
    status: Status
    ranked: list[tuple[int, float]]  # (answer_id, score), best first
    latency_ms: float
    entity_tokens: list[str] = field(default_factory=list)
    kind: str = ""  # free label, e.g. realistic-set category

    @property
    def rank(self) -> int | None:
        """1-based rank of the gold answer, or None if it was not returned."""
        for i, (answer_id, _) in enumerate(self.ranked, start=1):
            if answer_id == self.gold:
                return i
        return None

    @property
    def top1(self) -> int | None:
        return self.ranked[0][0] if self.ranked else None

    @property
    def confidence(self) -> float:
        return self.ranked[0][1] if self.ranked else 0.0


def top_k_accuracy(records: list[QueryRecord], k: int) -> float:
    return _mean([r.rank is not None and r.rank <= k for r in records])


def mrr(records: list[QueryRecord], cutoff: int = 10) -> float:
    return _mean([1 / r.rank if r.rank is not None and r.rank <= cutoff else 0.0 for r in records])


def summarize(records: list[QueryRecord]) -> dict:
    """Ranking quality, decision quality and latency for queries that have a gold answer."""
    if not records:
        return {"n": 0}
    answered = [r for r in records if r.status == "answer"]
    suggested = [r for r in records if r.status == "suggest"]
    latencies = [r.latency_ms for r in records]
    return {
        "n": len(records),
        "top1": top_k_accuracy(records, 1),
        "top3": top_k_accuracy(records, 3),
        "mrr10": mrr(records, 10),
        "coverage": len(answered) / len(records),
        "precision_at_answer": _mean([r.top1 == r.gold for r in answered]) if answered else None,
        "suggest_rate": len(suggested) / len(records),
        "suggest_hit_at3": _mean([r.rank is not None and r.rank <= 3 for r in suggested])
        if suggested
        else None,
        "not_found_rate": _mean([r.status == "not_found" for r in records]),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }


def status_rates(records: list[QueryRecord]) -> dict:
    """Answer / suggest / not_found shares, for query sets without a gold answer."""
    n = len(records)
    if not n:
        return {"n": 0}
    return {"n": n} | {
        f"{status}_rate": sum(r.status == status for r in records) / n
        for status in ("answer", "suggest", "not_found")
    }


def _mean(values: list) -> float:
    return float(np.mean(values)) if values else 0.0
