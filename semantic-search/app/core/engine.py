"""QAEngine: preprocessing → retrievers → answer aggregation → RRF fusion → decision.

QAEngine and the evaluation baselines both return QAResult, so the evaluation code does not
care which system produced it.
"""

import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

from app.core.config import Settings
from app.core.data import Corpus
from app.core.entities import Entity, EntityExtractor, TaggedText
from app.core.fusion import max_rrf_score, weighted_rrf
from app.core.normalize import strip_phrases
from app.core.ranker import aggregate_max
from app.core.retrievers.base import Retriever
from app.core.retrievers.bm25 import BM25Retriever
from app.core.retrievers.tfidf_char import CharTfidfRetriever

Status = Literal["answer", "suggest", "not_found"]


@dataclass(frozen=True)
class Candidate:
    answer_id: int
    score: float


@dataclass(frozen=True)
class QAResult:
    status: Status
    candidates: list[Candidate]  # best first
    entities: list[Entity] = field(default_factory=list)
    debug: dict | None = None
    latency_ms: float = 0.0

    @property
    def top(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def confidence(self) -> float:
        return self.candidates[0].score if self.candidates else 0.0


class QASystem(Protocol):
    def ask(self, query: str, k: int = 10) -> QAResult: ...


def decide_status(
    candidates: list[Candidate],
    answer_threshold: float,
    suggest_threshold: float,
    min_margin: float = 0.0,
) -> Status:
    """Answer only when the top score is high and clearly ahead of the runner-up."""
    if not candidates or candidates[0].score < suggest_threshold:
        return "not_found"
    margin = candidates[0].score - (candidates[1].score if len(candidates) > 1 else 0.0)
    if candidates[0].score >= answer_threshold and margin >= min_margin:
        return "answer"
    return "suggest"


@dataclass
class _Channel:
    """One retriever over one document set (questions or answers)."""

    name: str
    retriever: Retriever
    doc_answer_ids: list[int]
    weight: float


class QAEngine:
    """In-memory search engine over a Corpus. Build once, then call ask() per query."""

    def __init__(self, corpus: Corpus, settings: Settings) -> None:
        start = time.perf_counter()
        self.corpus = corpus
        self.settings = settings
        self.extractor = EntityExtractor.from_settings(settings)

        answer_ids = sorted(corpus.answers)
        doc_sets = [
            (
                "questions",
                [self._preprocess(q.text).text for q in corpus.questions],
                [q.answer_id for q in corpus.questions],
                1.0,
            )
        ]
        if settings.use_answer_index:
            doc_sets.append(
                (
                    "answers",
                    [self._preprocess(corpus.answers[a].text).text for a in answer_ids],
                    answer_ids,
                    settings.answer_index_weight,
                )
            )

        self.channels: list[_Channel] = []
        for set_name, docs, doc_answer_ids, set_weight in doc_sets:
            if not docs:
                continue
            for kind, weight, make in self._retriever_specs():
                retriever = make()
                retriever.fit(docs)
                self.channels.append(
                    _Channel(f"{kind}:{set_name}", retriever, doc_answer_ids, weight * set_weight)
                )
        if not self.channels:
            raise ValueError("No retriever enabled: set USE_TFIDF and/or USE_BM25")
        self.build_seconds = time.perf_counter() - start

    def _retriever_specs(self):
        s = self.settings
        if s.use_tfidf:
            yield (
                "tfidf",
                s.weight_tfidf,
                lambda: CharTfidfRetriever(
                    (s.tfidf_ngram_min, s.tfidf_ngram_max), s.tfidf_entity_weight
                ),
            )
        if s.use_bm25:
            yield "bm25", s.weight_bm25, lambda: BM25Retriever(s.bm25_k1, s.bm25_b)

    def _preprocess(self, text: str) -> TaggedText:
        tagged = self.extractor.process(text)
        if self.settings.strip_template_fillers:
            return TaggedText(
                strip_phrases(tagged.text, self.settings.template_fillers), tagged.entities
            )
        return tagged

    def ask(self, query: str, k: int = 3, debug: bool = False) -> QAResult:
        start = time.perf_counter()
        s = self.settings
        tagged = self._preprocess(query)

        ranked_lists: list[tuple[float, list[int]]] = []
        per_channel: dict[str, dict[int, tuple[int, float]]] = {}
        for ch in self.channels:
            ranked = aggregate_max(
                ch.retriever.search(tagged.text, s.candidate_pool), ch.doc_answer_ids
            )
            ranked_lists.append((ch.weight, [answer_id for answer_id, _ in ranked]))
            if debug:
                per_channel[ch.name] = {a: (rank, sc) for rank, (a, sc) in enumerate(ranked, 1)}

        fused = weighted_rrf(ranked_lists, s.rrf_k)
        scale = max_rrf_score([w for w, _ in ranked_lists], s.rrf_k)
        best = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
        candidates = [Candidate(answer_id, score / scale) for answer_id, score in best]
        status = decide_status(
            candidates, s.answer_threshold, s.suggest_threshold, s.answer_min_margin
        )

        debug_info = None
        if debug:
            debug_info = {
                "processed_query": tagged.text,
                "channels": {
                    name: {
                        str(c.answer_id): {
                            "rank": hits[c.answer_id][0],
                            "score": hits[c.answer_id][1],
                        }
                        for c in candidates
                        if c.answer_id in hits
                    }
                    for name, hits in per_channel.items()
                },
            }
        return QAResult(
            status=status,
            candidates=candidates,
            entities=tagged.entities,
            debug=debug_info,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
