"""BM25 over whitespace tokens of the tagged text, so entity tokens stay whole."""

import bm25s

from app.core.retrievers.base import top_k


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.model = bm25s.BM25(k1=k1, b=b)

    def fit(self, docs: list[str]) -> None:
        self.model.index([d.split() for d in docs], show_progress=False)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        tokens = query.split()
        if not tokens:
            return []
        return top_k(self.model.get_scores(tokens), k)
