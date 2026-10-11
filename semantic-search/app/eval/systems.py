"""Systems the evaluation can run. Each factory builds a QASystem from a corpus.

Ablation rows are added here as retrievers land (Phase 4+).
"""

import random
from collections.abc import Callable

from app.core.config import Settings
from app.core.data import Corpus
from app.core.engine import Candidate, QAResult, QASystem, decide_status


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
            self.settings.answer_threshold,
            self.settings.suggest_threshold,
            self.settings.answer_min_margin,
        )
        return QAResult(status=status, candidates=candidates)


SystemFactory = Callable[[Corpus, Settings], QASystem]

SYSTEMS: dict[str, SystemFactory] = {
    "random": RandomSystem,
}
