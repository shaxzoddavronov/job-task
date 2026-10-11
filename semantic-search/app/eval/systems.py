"""Systems the evaluation can run. Each factory builds a QASystem from a corpus.

Ablation rows are QAEngine with settings overrides; `engine` is the production config (the
current settings), so it should match the best row once defaults are updated.
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
            self.settings.answer_threshold,
            self.settings.suggest_threshold,
            self.settings.answer_min_margin,
        )
        return QAResult(status=status, candidates=candidates)


SystemFactory = Callable[[Corpus, Settings], QASystem]


def engine_with(**overrides) -> SystemFactory:
    """Factory for a QAEngine whose settings differ from the current ones by `overrides`."""

    def factory(corpus: Corpus, settings: Settings) -> QASystem:
        return QAEngine(corpus, settings.model_copy(update=overrides))

    return factory


# Plan rows 1-4 pin the retrievers and the answer index; everything else follows the settings.
_PLAN_ROW = {"strip_template_fillers": False, "normalize_strip_apostrophes": False}

SYSTEMS: dict[str, SystemFactory] = {
    "random": RandomSystem,
    "1 tfidf": engine_with(**_PLAN_ROW, use_tfidf=True, use_bm25=False, use_answer_index=False),
    "2 bm25": engine_with(**_PLAN_ROW, use_tfidf=False, use_bm25=True, use_answer_index=False),
    "3 tfidf+bm25": engine_with(**_PLAN_ROW, use_tfidf=True, use_bm25=True, use_answer_index=False),
    "4 tfidf+bm25+answers": engine_with(
        **_PLAN_ROW, use_tfidf=True, use_bm25=True, use_answer_index=True
    ),
    # E4 and apostrophe stripping, measured on top of the production config
    "4a engine -fillers": engine_with(strip_template_fillers=True),
    "4b engine -apostrophes": engine_with(normalize_strip_apostrophes=True),
    "engine": QAEngine,
}
