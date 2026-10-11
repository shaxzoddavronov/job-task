import numpy as np
import pytest

from app.core.ranker import (
    FEATURES,
    CandidateFeatures,
    category_agreement,
    entity_conflict,
    entity_match,
    rerank,
    score_matrix,
)


def weights(**values):
    w = np.zeros(len(FEATURES))
    for name, value in values.items():
        w[FEATURES.index(name)] = value
    return w


def test_entity_match_fraction():
    assert entity_match({"ORG_dxm", "TERM_eri"}, {"ORG_dxm"}) == 0.5
    assert entity_match(set(), {"ORG_dxm"}) == 0.0


@pytest.mark.parametrize(
    ("query", "candidate", "expected"),
    [
        ({"LAW_pq_4567"}, {"LAW_pq_4576"}, 1.0),  # same type, other number
        ({"LAW_pq_4567"}, {"LAW_pq_4567", "LAW_pq_4576"}, 0.0),  # cites the right one too
        ({"LAW_pq_4567"}, {"LAW_pf_4576"}, 0.0),  # different law type
        ({"LAW_pq_4567"}, set(), 0.0),
        ({"ART_15"}, {"ART_16"}, 1.0),
        ({"ORG_dxm"}, {"ORG_iiv"}, 0.0),  # organizations never conflict
    ],
)
def test_entity_conflict(query, candidate, expected):
    assert entity_conflict(query, candidate) == expected


def test_category_agreement_majority_of_top():
    assert category_agreement([1, 2, 1, 1, 3, 2], top=5) == [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert category_agreement([None, None]) == [0.0, 0.0]


def test_score_is_weighted_mean_in_unit_interval():
    f = np.zeros((1, len(FEATURES)))
    f[0, FEATURES.index("dense")] = 1.0
    f[0, FEATURES.index("rerank")] = 0.5
    assert score_matrix(f, weights(dense=1, rerank=1), True)[0] == pytest.approx(0.75)


def test_entity_weight_dropped_without_query_entities():
    f = np.zeros((1, len(FEATURES)))
    f[0, FEATURES.index("dense")] = 1.0
    w = weights(dense=1, entity_match=1)
    assert score_matrix(f, w, False)[0] == pytest.approx(1.0)  # not capped at 0.5
    assert score_matrix(f, w, True)[0] == pytest.approx(0.5)


def test_conflict_penalty_and_clipping():
    f = np.zeros((1, len(FEATURES)))
    f[0, FEATURES.index("dense")] = 1.0
    f[0, FEATURES.index("entity_conflict")] = 1.0
    assert score_matrix(f, weights(dense=1, entity_conflict=3), True)[0] == 0.0


def test_batch_matches_single():
    rng = np.random.default_rng(0)
    batch = rng.random((3, 4, len(FEATURES)))
    flags = np.array([True, False, True])
    w = weights(dense=1, sim_a=2, entity_match=1, entity_conflict=1)
    expected = [score_matrix(batch[i], w, flags[i]) for i in range(3)]
    assert np.allclose(score_matrix(batch, w, flags), np.stack(expected))


def test_rerank_orders_and_breaks_ties_by_fused_order():
    cands = [
        CandidateFeatures(1, {"dense": 1.0, "rerank": 0.2}),
        CandidateFeatures(2, {"dense": 0.5, "rerank": 1.0}),
        CandidateFeatures(3, {"dense": 0.5, "rerank": 1.0}),
    ]
    assert [a for a, _ in rerank(cands, weights(dense=1, rerank=1), False)] == [2, 3, 1]
    assert rerank([], weights(dense=1), False) == []


def test_top_k_drops_zero_scores_and_sorts():
    from app.core.retrievers.base import top_k

    assert top_k(np.array([0.0, 0.5, 0.2, 0.0]), 3) == [(1, 0.5), (2, 0.2)]
    assert top_k(np.array([]), 3) == []


def test_aggregate_max_keeps_best_score_per_answer():
    from app.core.ranker import aggregate_max

    hits = [(0, 0.9), (1, 0.8), (2, 0.95), (3, 0.1)]
    assert aggregate_max(hits, [1, 1, 2, 3]) == [(2, 0.95), (1, 0.9), (3, 0.1)]
