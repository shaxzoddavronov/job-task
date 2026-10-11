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


# Earlier rows pin the settings of their phase so their numbers stay reproducible as the
# defaults move on.
_PHASE4 = {"use_reranker": False, "use_synonyms": False, "use_static_emb": False}
_PLAN_ROW = _PHASE4 | {"strip_template_fillers": False, "normalize_strip_apostrophes": False}
_PHASE4_ENGINE = _PHASE4 | {"use_tfidf": True, "use_bm25": False, "use_answer_index": True}
_PHASE5_WEIGHTS = {
    "rerank_w_rrf": 0.5,
    "rerank_w_sim_q": 2.0,
    "rerank_w_sim_a": 0.0,
    "rerank_w_sim_e": 0.0,
    "rerank_w_fuzzy": 2.0,
    "rerank_w_entity_match": 0.5,
    "rerank_w_entity_match_q": 0.0,
    "rerank_w_category": 0.5,
    "rerank_w_entity_conflict": 3.0,
}
_PHASE5 = _PHASE5_WEIGHTS | {"use_reranker": True, "use_static_emb": False}
# Option B tuned on its own (scripts.evaluate --tune-reranker with the distilled model)
_DISTILLED = {
    "use_static_emb": True,
    "static_model_path": "models/bge-m3-static-256",
    "weight_static": 1.0,
    "rerank_w_rrf": 1.0,
    "rerank_w_sim_q": 0.0,
    "rerank_w_sim_a": 0.5,
    "rerank_w_sim_e": 0.5,
    "rerank_w_fuzzy": 2.0,
    "rerank_w_entity_match": 0.0,
    "rerank_w_entity_match_q": 0.0,
    "rerank_w_category": 0.0,
    "rerank_w_entity_conflict": 3.0,
}

SYSTEMS: dict[str, SystemFactory] = {
    "random": RandomSystem,
    "1 tfidf": engine_with(**_PLAN_ROW, use_tfidf=True, use_bm25=False, use_answer_index=False),
    "2 bm25": engine_with(**_PLAN_ROW, use_tfidf=False, use_bm25=True, use_answer_index=False),
    "3 tfidf+bm25": engine_with(**_PLAN_ROW, use_tfidf=True, use_bm25=True, use_answer_index=False),
    "4 tfidf+bm25+answers": engine_with(
        **_PLAN_ROW, use_tfidf=True, use_bm25=True, use_answer_index=True
    ),
    "4* phase-4 engine": engine_with(**_PHASE4_ENGINE),
    "4a phase-4 -fillers": engine_with(**_PHASE4_ENGINE, strip_template_fillers=True),
    "4b phase-4 -apostrophes": engine_with(**_PHASE4_ENGINE, normalize_strip_apostrophes=True),
    "5 +rerank": engine_with(**_PHASE5, use_synonyms=False),
    "5a +rerank -entity": engine_with(
        **(_PHASE5 | {"rerank_w_entity_match": 0.0, "rerank_w_entity_conflict": 0.0}),
        use_synonyms=False,
    ),
    "6 +rerank +synonyms": engine_with(**_PHASE5, use_synonyms=True),
    "6b +rerank +synonyms +bm25": engine_with(**_PHASE5, use_synonyms=True, use_bm25=True),
    "7a +model2vec pretrained": engine_with(
        use_static_emb=True, static_model_path="models/potion-multilingual-128M"
    ),
    "7b +model2vec distilled bge-m3": engine_with(**_DISTILLED),
    "engine": QAEngine,
}
