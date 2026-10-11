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
    f[0, FEATURES.index("rrf")] = 1.0
    f[0, FEATURES.index("fuzzy")] = 0.5
    assert score_matrix(f, weights(rrf=1, fuzzy=1), True)[0] == pytest.approx(0.75)


def test_entity_weight_dropped_without_query_entities():
    f = np.zeros((1, len(FEATURES)))
    f[0, FEATURES.index("rrf")] = 1.0
    w = weights(rrf=1, entity_match=1)
    assert score_matrix(f, w, False)[0] == pytest.approx(1.0)  # not capped at 0.5
    assert score_matrix(f, w, True)[0] == pytest.approx(0.5)


def test_conflict_penalty_and_clipping():
    f = np.zeros((1, len(FEATURES)))
    f[0, FEATURES.index("rrf")] = 1.0
    f[0, FEATURES.index("entity_conflict")] = 1.0
    assert score_matrix(f, weights(rrf=1, entity_conflict=3), True)[0] == 0.0


def test_batch_matches_single():
    rng = np.random.default_rng(0)
    batch = rng.random((3, 4, len(FEATURES)))
    flags = np.array([True, False, True])
    w = weights(rrf=1, sim_q=2, entity_match=1, entity_conflict=1)
    expected = [score_matrix(batch[i], w, flags[i]) for i in range(3)]
    assert np.allclose(score_matrix(batch, w, flags), np.stack(expected))


def test_rerank_orders_and_breaks_ties_by_fused_order():
    cands = [
        CandidateFeatures(1, {"rrf": 1.0, "fuzzy": 0.2}),
        CandidateFeatures(2, {"rrf": 0.5, "fuzzy": 1.0}),
        CandidateFeatures(3, {"rrf": 0.5, "fuzzy": 1.0}),
    ]
    assert [a for a, _ in rerank(cands, weights(rrf=1, fuzzy=1), False)] == [2, 3, 1]
    assert rerank([], weights(rrf=1), False) == []
