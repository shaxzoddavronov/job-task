"""Char n-gram TF-IDF with a separate exact-match channel for entity tokens.

Char n-grams handle Uzbek morphology and typos but would blur LAW_pq_4567 and LAW_pq_4576,
so entity tokens are removed from the char text and scored by a word-level vectorizer:
score = cos(char) + entity_weight * cos(entities).
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.normalize import ENTITY_TOKEN_RE
from app.core.retrievers.base import top_k


def _plain(text: str) -> str:
    return ENTITY_TOKEN_RE.sub(" ", text)


def _entities(text: str) -> str:
    return " ".join(ENTITY_TOKEN_RE.findall(text))


class CharTfidfRetriever:
    def __init__(self, ngram_range: tuple[int, int] = (3, 5), entity_weight: float = 0.5):
        self.entity_weight = entity_weight
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, sublinear_tf=True)
        self.ent = TfidfVectorizer(
            token_pattern=ENTITY_TOKEN_RE.pattern, lowercase=False, sublinear_tf=True
        )
        self._char_matrix = None
        self._ent_matrix = None

    def fit(self, docs: list[str]) -> None:
        self._char_matrix = self.char.fit_transform([_plain(d) for d in docs])
        entity_docs = [_entities(d) for d in docs]
        self._ent_matrix = self.ent.fit_transform(entity_docs) if any(entity_docs) else None

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = (self._char_matrix @ self.char.transform([_plain(query)]).T).toarray().ravel()
        if self._ent_matrix is not None and self.entity_weight and _entities(query):
            ent_q = self.ent.transform([_entities(query)])
            scores = scores + self.entity_weight * (self._ent_matrix @ ent_q.T).toarray().ravel()
        return top_k(np.asarray(scores), k)
