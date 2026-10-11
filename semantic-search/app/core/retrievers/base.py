"""Retriever protocol shared by all retrievers. Inputs are entity-tagged, cleaned texts."""

from typing import Protocol

import numpy as np


class Retriever(Protocol):
    def fit(self, docs: list[str]) -> None: ...

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-k (doc_index, score) pairs, best first; documents scoring 0 are dropped."""
        ...


def top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Indices and values of the k largest positive scores, best first."""
    if k <= 0 or scores.size == 0:
        return []
    k = min(k, scores.size)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]
