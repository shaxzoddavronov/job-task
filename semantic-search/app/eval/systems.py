"""Systems the evaluation can run. Each factory builds a QASystem from a corpus.

Rows are QAEngine with settings overrides; `engine` is the production config (the current
settings: rerank mode on a GPU, dense mode on CPU).
"""

import random
from collections.abc import Callable

from app.core.config import Settings
from app.core.data import Corpus
from app.core.engine import Candidate, QAEngine, QAResult, QASystem, decide_status


class RandomSystem:
    """Dummy baseline: random ranking with random scores. Sanity floor for every metric."""

    def __init__(self, corpus: Corpus, settings: Settings) -> None:
        self.answer_ids = sorted(corpus.answers)
        self.settings = settings
        self.rng = random.Random(settings.eval_seed)

    def ask(self, query: str, k: int = 10) -> QAResult:
        ids = self.rng.sample(self.answer_ids, min(k, len(self.answer_ids)))
        scores = sorted((self.rng.random() for _ in ids), reverse=True)
        candidates = [Candidate(a, s) for a, s in zip(ids, scores, strict=True)]
        status = decide_status(
            candidates,
            self.settings.answer_threshold_dense,
            self.settings.suggest_threshold_dense,
            self.settings.answer_min_margin_dense,
        )
        return QAResult(status=status, candidates=candidates)


SystemFactory = Callable[[Corpus, Settings], QASystem]


def engine_with(**overrides) -> SystemFactory:
    """Factory for a QAEngine whose settings differ from the current ones by `overrides`."""

    def factory(corpus: Corpus, settings: Settings) -> QASystem:
        return QAEngine(corpus, settings.model_copy(update=overrides))

    return factory


SYSTEMS: dict[str, SystemFactory] = {
    "random": RandomSystem,
    "dense (bge-m3)": engine_with(reranker="off"),
    "dense + rerank (bge-reranker-v2-m3)": engine_with(reranker="on"),
    "engine": QAEngine,
}
