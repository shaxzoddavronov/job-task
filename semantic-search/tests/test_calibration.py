import numpy as np
import pytest

from app.core.engine import Candidate, decide_status
from app.eval.calibration import ScoreTable, calibrate, decide, decision_metrics
from app.eval.metrics import QueryRecord


def table(s1, s2, correct, hit3=None):
    return ScoreTable(
        np.array(s1, float),
        np.array(s2, float),
        np.array(correct, bool),
        np.array(hit3 if hit3 is not None else correct, bool),
    )


@pytest.mark.parametrize("margin", [0.0, 0.1])
def test_offline_decision_matches_engine_rule(margin):
    rng = np.random.default_rng(1)
    s1 = rng.random(200)
    s2 = s1 * rng.random(200)
    t = table(s1, s2, [True] * 200)
    answered, suggested, not_found = decide(t, 0.6, 0.3, margin)
    for i in range(200):
        status = decide_status([Candidate(1, s1[i]), Candidate(2, s2[i])], 0.6, 0.3, margin)
        expected = "answer" if answered[i] else "suggest" if suggested[i] else "not_found"
        assert status == expected


def test_score_table_from_records():
    records = [
        QueryRecord("q", 1, "answer", [(1, 0.9), (2, 0.5)], 1.0),
        QueryRecord("q", 3, "suggest", [(1, 0.6), (2, 0.5), (3, 0.4)], 1.0),
        QueryRecord("q", None, "not_found", [(1, 0.2)], 1.0),
    ]
    t = ScoreTable.from_records(records)
    assert t.s1.tolist() == [0.9, 0.6, 0.2] and t.s2.tolist() == [0.5, 0.5, 0.0]
    assert t.correct.tolist() == [True, False, False] and t.hit3.tolist() == [True, True, False]


def test_metrics_useful_counts_suggestions_with_gold_in_top3():
    t = table([0.9, 0.5, 0.1], [0.1, 0.4, 0.0], [True, False, False], [True, True, False])
    m = decision_metrics(t, answer=0.8, suggest=0.3, margin=0.0)
    assert m["coverage"] == pytest.approx(1 / 3) and m["precision_at_answer"] == 1.0
    assert m["useful"] == pytest.approx(2 / 3) and m["not_found_rate"] == pytest.approx(1 / 3)


def test_calibrate_respects_precision_and_ood_constraints():
    # dev: high scores are right, mid scores wrong; OOD sits in the middle.
    dev = table([0.9, 0.85, 0.8, 0.6, 0.55], [0.1] * 5, [1, 1, 1, 0, 0], [1, 1, 1, 1, 0])
    val = table([0.95] * 5, [0.1] * 5, [1] * 5)
    ood = table([0.5, 0.45, 0.3, 0.2, 0.1], [0.1] * 5, [0] * 5)
    grid = np.round(np.arange(0.3, 0.96, 0.05), 2)
    result = calibrate(dev, val, ood, grid, [0.0], min_ood_not_found=0.8)
    assert result.answer_threshold == pytest.approx(0.65)  # lowest that keeps 100% precision
    assert result.dev["precision_at_answer"] == 1.0
    assert result.ood["not_found_rate"] >= 0.8
    assert result.suggest_threshold <= result.answer_threshold


def test_calibrate_returns_none_when_impossible():
    dev = table([0.9], [0.1], [False])
    assert calibrate(dev, dev, dev, np.array([0.5]), [0.0]) is None
