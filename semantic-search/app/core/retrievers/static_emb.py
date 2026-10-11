"""Static-embedding retriever (Model2Vec): a numpy lookup table, no transformer at inference.

Entity tokens are turned back into words before encoding (TERM_id_karta → "id karta"),
because the subword tokenizer would otherwise split them into noise.
"""

from pathlib import Path

import numpy as np

from app.core.normalize import ENTITY_TOKEN_RE
from app.core.retrievers.base import top_k


def readable(text: str) -> str:
    return ENTITY_TOKEN_RE.sub(lambda m: m.group(0).split("_", 1)[1].replace("_", " "), text)


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-9)


class StaticEmbeddingRetriever:
    _cache: dict[str, object] = {}  # one model per path per process (loading takes seconds)

    def __init__(self, model_path: Path) -> None:
        from model2vec import StaticModel  # imported lazily: optional component

        key = str(model_path)
        if key not in self._cache:
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Static embedding model not found at {model_path}; see README (make distill)"
                )
            self._cache[key] = StaticModel.from_pretrained(key)
        self.model = self._cache[key]
        self._matrix: np.ndarray | None = None

    def fit(self, docs: list[str]) -> None:
        self._matrix = _unit_rows(np.asarray(self.model.encode([readable(d) for d in docs])))

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not query.strip():
            return []
        q = _unit_rows(np.asarray(self.model.encode([readable(query)])))[0]
        return top_k(self._matrix @ q, k)
