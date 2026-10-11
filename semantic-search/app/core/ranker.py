"""From document hits to answer candidates, and the final linear score over model and rule features.

Every candidate gets features in [0, 1]; the re-ranker score is a weighted mean of the positive
features minus a conflict penalty, so it doubles as an absolute confidence:
    score = (Σ wᵢ·fᵢ − w_conflict·entity_conflict) / Σ wᵢ, clipped to [0, 1].
The entity weights are left out (numerator and denominator) when the query has no entities.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from app.core.config import Settings

# Order matters: score_matrix and the tuner index features by position.
POSITIVE_FEATURES = (
    "rerank",  # cross-encoder relevance (GPU mode)
    "dense",  # bi-encoder similarity: best question + answer-text bonus
    "sim_a",  # bi-encoder similarity to the answer text alone
    "entity_match",
    "entity_match_q",
    "category",
)
FEATURES = (*POSITIVE_FEATURES, "entity_conflict")
_ENTITY_IDX = [FEATURES.index("entity_match"), FEATURES.index("entity_match_q")]
_CONFLICT_IDX = FEATURES.index("entity_conflict")


@dataclass
class CandidateFeatures:
    answer_id: int
    features: dict[str, float]

    def vector(self) -> np.ndarray:
        return np.array([self.features.get(name, 0.0) for name in FEATURES])


def weights_from_settings(settings: Settings, mode: str) -> np.ndarray:
    """Weight vector for mode "rerank" (with cross-encoder) or "dense" (without)."""
    table = settings.weights_rerank if mode == "rerank" else settings.weights_dense
    unknown = set(table) - set(FEATURES)
    if unknown:
        raise ValueError(f"Unknown re-ranker features {sorted(unknown)}; known: {FEATURES}")
    return np.array([float(table.get(name, 0.0)) for name in FEATURES])


def aggregate_max(
    hits: list[tuple[int, float]], doc_answer_ids: list[int]
) -> list[tuple[int, float]]:
    """Map document hits to answer ids, keep each answer's best score, sort best first."""
    best: dict[int, float] = {}
    for doc_idx, score in hits:
        answer_id = doc_answer_ids[doc_idx]
        if score > best.get(answer_id, float("-inf")):
            best[answer_id] = score
    return sorted(best.items(), key=lambda item: -item[1])


# --- Features --------------------------------------------------------------------------------


def entity_match(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    """Share of the query's entity tokens that the candidate (questions or answer) mentions."""
    if not query_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens)


def _numbered(tokens: set[str]) -> dict[str, set[str]]:
    """Group LAW/ART tokens by kind: LAW_pq_4567 → {"LAW_pq": {"4567"}}, ART_15 → {"ART": ...}."""
    groups: dict[str, set[str]] = {}
    for token in tokens:
        if token.startswith("LAW_") or token.startswith("ART_"):
            kind, _, number = token.rpartition("_")
            groups.setdefault(kind, set()).add(number)
    return groups


def entity_conflict(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    """1.0 if the candidate cites the same kind of law/article with a different number only."""
    candidate = _numbered(candidate_tokens)
    for kind, numbers in _numbered(query_tokens).items():
        cited = candidate.get(kind, set())
        if cited and not (cited & numbers):
            return 1.0
    return 0.0


def category_agreement(categories: list[int | None], top: int = 5) -> list[float]:
    """1.0 for candidates whose category is the most common one among the first `top`."""
    known = [c for c in categories[:top] if c is not None]
    if not known:
        return [0.0] * len(categories)
    majority = Counter(known).most_common(1)[0][0]
    return [1.0 if c == majority else 0.0 for c in categories]


# --- Scoring ---------------------------------------------------------------------------------


def score_matrix(features: np.ndarray, weights: np.ndarray, has_entities: np.ndarray) -> np.ndarray:
    """Linear re-ranker scores for features of shape (..., n_candidates, n_features).

    `has_entities` has the leading shape (one flag per query). Works for a single query
    (features 2-D, has_entities scalar) and for a padded batch (features 3-D).
    """
    positive = weights[:_CONFLICT_IDX]
    numerator = features[..., :_CONFLICT_IDX] @ positive
    numerator = numerator - features[..., _CONFLICT_IDX] * weights[_CONFLICT_IDX]
    has_entities = np.asarray(has_entities, dtype=float)
    denominator = positive.sum() - weights[_ENTITY_IDX].sum() * (1.0 - has_entities)
    denominator = np.where(denominator > 0, denominator, np.inf)
    return np.clip(numerator / denominator[..., None], 0.0, 1.0)


def rerank(
    candidates: list[CandidateFeatures], weights: np.ndarray, has_entities: bool
) -> list[tuple[int, float]]:
    """Candidates re-scored and sorted best first (stable: ties keep the fused order)."""
    if not candidates:
        return []
    matrix = np.stack([c.vector() for c in candidates])
    scores = score_matrix(matrix, weights, has_entities)
    order = np.argsort(-scores, kind="stable")
    return [(candidates[i].answer_id, float(scores[i])) for i in order]
