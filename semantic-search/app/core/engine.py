"""QAEngine: preprocessing → retrievers → aggregation → RRF fusion → re-ranking → decision.

QAEngine and the evaluation baselines both return QAResult, so the evaluation code does not
care which system produced it.
"""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Literal, Protocol

from rapidfuzz import fuzz

from app.core.config import Settings
from app.core.data import Corpus
from app.core.entities import Entity, EntityExtractor, TaggedText
from app.core.fusion import max_rrf_score, weighted_rrf
from app.core.normalize import strip_phrases
from app.core.ranker import (
    CandidateFeatures,
    aggregate_max,
    category_agreement,
    entity_conflict,
    entity_match,
    rerank,
    weights_from_settings,
)
from app.core.retrievers.base import Retriever
from app.core.retrievers.bm25 import BM25Retriever
from app.core.retrievers.static_emb import StaticEmbeddingRetriever
from app.core.retrievers.tfidf_char import CharTfidfRetriever
from app.core.synonyms import QueryExpander

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
    reason: str | None = None  # e.g. "unknown_law" when status is forced to not_found

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


@dataclass
class Retrieval:
    """First-stage output: candidates with re-ranker features, best fused first."""

    tagged: TaggedText
    retrieval_text: str
    candidates: list[CandidateFeatures]
    channel_hits: dict[str, dict[int, tuple[int, float]]]


# Entity kinds that must exist in the data; a query citing an unknown one gets not_found.
_MUST_EXIST = ("LAW_", "ART_")


class QAEngine:
    """In-memory search engine over a Corpus. Build once, then call ask() per query."""

    def __init__(self, corpus: Corpus, settings: Settings) -> None:
        start = time.perf_counter()
        self.corpus = corpus
        self.settings = settings
        self.extractor = EntityExtractor.from_settings(settings)
        self.expander = (
            QueryExpander.from_yaml(settings.synonyms_path)
            if settings.use_synonyms
            else QueryExpander([])
        )
        self.weights = weights_from_settings(settings)

        answer_ids = sorted(corpus.answers)
        question_tagged = [self._preprocess(q.text) for q in corpus.questions]
        answer_tagged = {a: self._preprocess(corpus.answers[a].text) for a in answer_ids}

        # Per-answer data for the re-ranker features
        self.answer_entities: dict[int, set[str]] = {
            a: set(t.tokens) for a, t in answer_tagged.items()
        }
        self.question_entities: dict[int, set[str]] = {a: set() for a in answer_ids}
        self.answer_questions: dict[int, list[str]] = {a: [] for a in answer_ids}
        for q, tagged in zip(corpus.questions, question_tagged, strict=True):
            self.answer_entities[q.answer_id] |= tagged.tokens
            self.question_entities[q.answer_id] |= tagged.tokens
            self.answer_questions[q.answer_id].append(tagged.text)
        self.known_entities = set().union(*self.answer_entities.values()) if answer_ids else set()

        doc_sets = [
            (
                "questions",
                [t.text for t in question_tagged],
                [q.answer_id for q in corpus.questions],
                1.0,
            )
        ]
        if settings.use_answer_index:
            doc_sets.append(
                (
                    "answers",
                    [answer_tagged[a].text for a in answer_ids],
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
            raise ValueError("No retriever enabled: set USE_TFIDF, USE_BM25 and/or USE_STATIC_EMB")
        self.build_seconds = time.perf_counter() - start

    def _retriever_specs(self) -> Iterator[tuple[str, float, Callable[[], Retriever]]]:
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
        if s.use_static_emb:
            yield "emb", s.weight_static, lambda: StaticEmbeddingRetriever(s.static_model_path)

    def _preprocess(self, text: str) -> TaggedText:
        tagged = self.extractor.process(text)
        if self.settings.strip_template_fillers:
            return TaggedText(
                strip_phrases(tagged.text, self.settings.template_fillers), tagged.entities
            )
        return tagged

    def unknown_entities(self, tagged: TaggedText) -> list[str]:
        """Law / article references in the query that appear nowhere in the data."""
        return sorted(
            t for t in tagged.tokens if t.startswith(_MUST_EXIST) and t not in self.known_entities
        )

    def retrieve(self, query: str) -> Retrieval:
        """Retrieval + fusion + features for the top `rerank_top_n` fused candidates."""
        s = self.settings
        tagged = self._preprocess(query)
        retrieval_text = self.expander.expand(tagged.text)

        ranked_lists: list[tuple[float, list[int]]] = []
        channel_hits: dict[str, dict[int, tuple[int, float]]] = {}
        for ch in self.channels:
            ranked = aggregate_max(
                ch.retriever.search(retrieval_text, s.candidate_pool), ch.doc_answer_ids
            )
            ranked_lists.append((ch.weight, [answer_id for answer_id, _ in ranked]))
            channel_hits[ch.name] = {a: (rank, sc) for rank, (a, sc) in enumerate(ranked, 1)}

        fused = weighted_rrf(ranked_lists, s.rrf_k)
        scale = max_rrf_score([w for w, _ in ranked_lists], s.rrf_k)
        top = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[: s.rerank_top_n]

        sim_scale = 1.0 + s.tfidf_entity_weight
        tfidf_q = channel_hits.get("tfidf:questions", {})
        tfidf_a = channel_hits.get("tfidf:answers", {})
        emb_q = channel_hits.get("emb:questions", {})
        query_tokens = tagged.tokens
        categories = [self.corpus.answers[a].category_id for a, _ in top]
        category_flags = category_agreement(categories)
        candidates = []
        for (answer_id, fused_score), category in zip(top, category_flags, strict=True):
            entities = self.answer_entities.get(answer_id, set())
            questions = self.answer_questions.get(answer_id, [])
            candidates.append(
                CandidateFeatures(
                    answer_id,
                    {
                        "rrf": fused_score / scale,
                        "sim_q": min(tfidf_q.get(answer_id, (0, 0.0))[1] / sim_scale, 1.0),
                        "sim_a": min(tfidf_a.get(answer_id, (0, 0.0))[1] / sim_scale, 1.0),
                        "sim_e": max(emb_q.get(answer_id, (0, 0.0))[1], 0.0),
                        "fuzzy": max(
                            (fuzz.token_set_ratio(tagged.text, q) for q in questions), default=0.0
                        )
                        / 100,
                        "entity_match": entity_match(query_tokens, entities),
                        "entity_match_q": entity_match(
                            query_tokens, self.question_entities.get(answer_id, set())
                        ),
                        "category": category,
                        "entity_conflict": entity_conflict(query_tokens, entities),
                    },
                )
            )
        return Retrieval(tagged, retrieval_text, candidates, channel_hits)

    def ask(self, query: str, k: int = 3, debug: bool = False) -> QAResult:
        start = time.perf_counter()
        s = self.settings
        r = self.retrieve(query)

        if s.use_reranker:
            scored = rerank(r.candidates, self.weights, bool(r.tagged.tokens))
        else:
            scored = [(c.answer_id, c.features["rrf"]) for c in r.candidates]
        candidates = [Candidate(answer_id, score) for answer_id, score in scored[:k]]
        status = decide_status(
            candidates, s.answer_threshold, s.suggest_threshold, s.answer_min_margin
        )
        unknown = self.unknown_entities(r.tagged)
        reason = None
        if unknown:
            status, reason = "not_found", "unknown_law"

        debug_info = None
        if debug:
            features = {c.answer_id: c.features for c in r.candidates}
            debug_info = {
                "processed_query": r.tagged.text,
                "retrieval_query": r.retrieval_text,
                "unknown_entities": unknown,
                "candidates": {
                    str(c.answer_id): {
                        "features": features[c.answer_id],
                        "channels": {
                            name: {"rank": hits[c.answer_id][0], "score": hits[c.answer_id][1]}
                            for name, hits in r.channel_hits.items()
                            if c.answer_id in hits
                        },
                    }
                    for c in candidates
                },
            }
        return QAResult(
            status=status,
            candidates=candidates,
            entities=r.tagged.entities,
            debug=debug_info,
            latency_ms=(time.perf_counter() - start) * 1000,
            reason=reason,
        )
