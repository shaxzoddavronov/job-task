"""Weighted Reciprocal Rank Fusion over ranked answer lists."""


def weighted_rrf(ranked_lists: list[tuple[float, list[int]]], k: int = 60) -> dict[int, float]:
    """score(a) = Σ wᵢ / (k + rankᵢ(a)), rank starting at 1; lists that miss `a` add nothing."""
    scores: dict[int, float] = {}
    for weight, items in ranked_lists:
        for rank, item in enumerate(items, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return scores


def max_rrf_score(weights: list[float], k: int = 60) -> float:
    """Score of an item ranked first in every list; used to scale fused scores to [0, 1]."""
    return sum(weights) / (k + 1)
