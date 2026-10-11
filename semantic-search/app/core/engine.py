"""QAEngine: normalize → entities → dense retrieval (BGE-M3) → cross-encoder rerank → decision.

QAEngine and the evaluation baselines both return QAResult, so the evaluation code does not
care which system produced it.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from app.core.config import Settings
from app.core.data import Corpus
from app.core.device import configure_cpu_threads, describe_device, resolve_device
from app.core.encoders import Encoder, Scorer, build_models
from app.core.entities import Entity, EntityExtractor, TaggedText
from app.core.normalize import normalize_script
from app.core.ranker import (
    CandidateFeatures,
    category_agreement,
    entity_conflict,
    entity_match,
    rerank,
    weights_from_settings,
)
from app.core.retrievers.dense import best_per_group, encode_cached
from app.core.synonyms import QueryExpander

log = logging.getLogger(__name__)

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
class Retrieval:
    """First-stage output: the top candidates with their features, best dense score first."""

    tagged: TaggedText
    retrieval_text: str
    candidates: list[CandidateFeatures]


# Entity kinds that must exist in the data; a query citing an unknown one gets not_found.
_MUST_EXIST = ("LAW_", "ART_")


class QAEngine:
    """In-memory search engine over a Corpus. Build once, then call ask() per query.

    Mode "rerank" (GPU by default): dense retrieval + cross-encoder. Mode "dense" (CPU by
    default): dense retrieval only. Each mode has its own weights and thresholds.
    """

    def __init__(
        self,
        corpus: Corpus,
        settings: Settings,
        encoder: Encoder | None = None,
        scorer: Scorer | None = None,
    ) -> None:
        start = time.perf_counter()
        self.corpus = corpus
        self.settings = settings
        self.device = "cpu" if settings.model_backend == "fake" else resolve_device(settings.device)
        if self.device == "cpu":
            configure_cpu_threads(settings.cpu_threads)
        with_reranker = settings.reranker == "on" or (
            settings.reranker == "auto" and self.device == "cuda"
        )
        if encoder is None:
            encoder, built_scorer = build_models(settings, self.device, with_reranker)
            scorer = scorer or built_scorer
        self.encoder = encoder
        self.scorer = scorer if with_reranker else None
        self.mode = "rerank" if self.scorer is not None else "dense"
        self.weights = weights_from_settings(settings, self.mode)
        self.thresholds = (
            getattr(settings, f"answer_threshold_{self.mode}"),
            getattr(settings, f"suggest_threshold_{self.mode}"),
            getattr(settings, f"answer_min_margin_{self.mode}"),
        )
        self.extractor = EntityExtractor.from_settings(settings)
        self.expander = (
            QueryExpander.from_yaml(settings.synonyms_path)
            if settings.use_synonyms
            else QueryExpander([])
        )
        self._lock = threading.Lock()  # one inference at a time (GPU memory, thread safety)

        # Answers and questions, aligned by position
        self.answer_ids = sorted(corpus.answers)
        position = {a: i for i, a in enumerate(self.answer_ids)}
        self.answer_texts = [normalize_script(corpus.answers[a].text) for a in self.answer_ids]
        self.question_texts = [normalize_script(q.text) for q in corpus.questions]
        self.question_group = np.array(
            [position[q.answer_id] for q in corpus.questions], dtype=np.int64
        )

        # Rule data per answer: entity tokens in its questions / answer text, known laws
        self.answer_entities: dict[int, set[str]] = {
            a: set(self.extractor.process(corpus.answers[a].text).tokens) for a in self.answer_ids
        }
        self.question_entities: dict[int, set[str]] = {a: set() for a in self.answer_ids}
        for q in corpus.questions:
            tokens = self.extractor.process(q.text).tokens
            self.answer_entities[q.answer_id] |= tokens
            self.question_entities[q.answer_id] |= tokens
        self.known_entities = (
            set().union(*self.answer_entities.values()) if self.answer_ids else set()
        )

        cache = settings.embedding_cache_dir
        self.question_vectors = encode_cached(self.encoder, self.question_texts, cache)
        self.answer_vectors = (
            encode_cached(self.encoder, self.answer_texts, cache)
            if settings.use_answer_index
            else None
        )
        self._warm_up()
        self.build_seconds = time.perf_counter() - start
        log.info(
            "Engine ready: mode=%s %s, %d answers, %d questions, built in %.1fs",
            self.mode,
            describe_device(self.device),
            len(self.answer_ids),
            len(self.question_texts),
            self.build_seconds,
        )

    def _warm_up(self) -> None:
        """First calls on a GPU are slow (kernel setup); pay that before serving."""
        if not self.answer_ids:
            return
        self.encoder.encode(["salom"])
        if self.scorer is not None:
            self.scorer.score([("salom", self.answer_texts[0])])

    @property
    def models(self) -> dict:
        s = self.settings
        return {
            "backend": s.model_backend,
            "device": self.device,
            "mode": self.mode,
            "bi_encoder": f"{s.bi_encoder_model}@{(s.bi_encoder_revision or 'latest')[:8]}",
            "reranker": f"{s.reranker_model}@{(s.reranker_revision or 'latest')[:8]}"
            if self.scorer is not None
            else None,
        }

    def unknown_entities(self, tagged: TaggedText) -> list[str]:
        """Law / article references in the query that appear nowhere in the data."""
        return sorted(
            t for t in tagged.tokens if t.startswith(_MUST_EXIST) and t not in self.known_entities
        )

    def _query_text(self, query: str, tagged: TaggedText) -> str:
        """Script-normalized query (case kept for the model) plus any synonym expansions."""
        text = normalize_script(query)
        expanded = self.expander.expand(tagged.text)
        extra = expanded[len(tagged.text) :].strip()
        return f"{text} {extra}" if extra else text

    def _pair_text(self, answer_pos: int, question_idx: int) -> str:
        question = self.question_texts[question_idx] if question_idx >= 0 else ""
        answer = self.answer_texts[answer_pos]
        if self.settings.rerank_pair == "answer":
            return answer
        if self.settings.rerank_pair == "question+answer":
            return f"{question} {answer}".strip()
        return question or answer

    def retrieve(self, query: str) -> Retrieval:
        """Dense retrieval + (in rerank mode) cross-encoder scores for the top candidates."""
        s = self.settings
        tagged = self.extractor.process(query)
        text = self._query_text(query, tagged)
        if not self.answer_ids or not text.strip():
            return Retrieval(tagged, text, [])

        with self._lock:
            query_vector = self.encoder.encode([text])[0]
        sim_q, best_q = best_per_group(
            self.question_vectors @ query_vector, self.question_group, len(self.answer_ids)
        )
        sim_q = np.where(np.isfinite(sim_q), sim_q, 0.0)
        sim_a = (
            self.answer_vectors @ query_vector
            if self.answer_vectors is not None
            else np.zeros(len(self.answer_ids), dtype=np.float32)
        )
        w = s.answer_index_weight if self.answer_vectors is not None else 0.0
        dense = (np.clip(sim_q, 0, 1) + w * np.clip(sim_a, 0, 1)) / (1.0 + w)
        top = np.argsort(-dense, kind="stable")[: s.rerank_top_n]

        rerank_scores = np.zeros(len(top), dtype=np.float32)
        if self.scorer is not None and len(top):
            pairs = [(text, self._pair_text(int(p), int(best_q[p]))) for p in top]
            with self._lock:
                rerank_scores = self.scorer.score(pairs)

        categories = [self.corpus.answers[self.answer_ids[p]].category_id for p in top]
        query_tokens = tagged.tokens
        candidates = []
        for p, rr, category in zip(top, rerank_scores, category_agreement(categories), strict=True):
            answer_id = self.answer_ids[int(p)]
            entities = self.answer_entities.get(answer_id, set())
            candidates.append(
                CandidateFeatures(
                    answer_id,
                    {
                        "rerank": float(rr),
                        "dense": float(dense[p]),
                        "sim_a": float(max(sim_a[p], 0.0)),
                        "entity_match": entity_match(query_tokens, entities),
                        "entity_match_q": entity_match(
                            query_tokens, self.question_entities.get(answer_id, set())
                        ),
                        "category": category,
                        "entity_conflict": entity_conflict(query_tokens, entities),
                    },
                )
            )
        return Retrieval(tagged, text, candidates)

    def ask(self, query: str, k: int = 3, debug: bool = False) -> QAResult:
        start = time.perf_counter()
        r = self.retrieve(query)
        scored = rerank(r.candidates, self.weights, bool(r.tagged.tokens))
        candidates = [Candidate(answer_id, score) for answer_id, score in scored[:k]]
        status = decide_status(candidates, *self.thresholds)
        unknown = self.unknown_entities(r.tagged)
        reason = None
        if unknown:
            status, reason = "not_found", "unknown_law"

        debug_info = None
        if debug:
            features = {c.answer_id: c.features for c in r.candidates}
            debug_info = {
                "mode": self.mode,
                "retrieval_query": r.retrieval_text,
                "unknown_entities": unknown,
                "candidates": {
                    str(c.answer_id): {"features": features[c.answer_id]} for c in candidates
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
