import pytest

from app.core.fusion import max_rrf_score, weighted_rrf
from app.core.ranker import aggregate_max


def test_rrf_values():
    scores = weighted_rrf([(1.0, [10, 20]), (1.0, [20, 30])], k=60)
    assert scores[10] == pytest.approx(1 / 61)
    assert scores[20] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[30] == pytest.approx(1 / 62)


def test_rrf_weights():
    scores = weighted_rrf([(1.0, [10]), (0.5, [20])], k=5)
    assert scores[10] == pytest.approx(1 / 6)
    assert scores[20] == pytest.approx(0.5 / 6)


def test_normalized_score_is_one_when_first_everywhere():
    weights = [1.0, 0.1]
    scores = weighted_rrf([(weights[0], [7, 8]), (weights[1], [7])], k=5)
    assert scores[7] / max_rrf_score(weights, k=5) == pytest.approx(1.0)
    assert scores[8] / max_rrf_score(weights, k=5) < 1.0


def test_aggregate_max_keeps_best_score_per_answer():
    hits = [(0, 0.9), (1, 0.8), (2, 0.95), (3, 0.1)]
    doc_answer_ids = [1, 1, 2, 3]
    assert aggregate_max(hits, doc_answer_ids) == [(2, 0.95), (1, 0.9), (3, 0.1)]
