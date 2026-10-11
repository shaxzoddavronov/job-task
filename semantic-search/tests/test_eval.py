import json

import pandas as pd
import pytest

from app.core.config import Settings
from app.core.data import Question, RawData, build_corpus
from app.core.engine import Candidate, decide_status
from app.eval.metrics import QueryRecord, mrr, summarize, top_k_accuracy
from app.eval.report import worst_failures, write_reports
from app.eval.runner import build_context, evaluate, law_injection_queries
from app.eval.split import split_questions
from app.eval.tuning import tune_reranker


def make_questions(counts: dict[int, int]) -> list[Question]:
    qs, qid = [], 1
    for answer_id, n in counts.items():
        for _ in range(n):
            qs.append(Question(qid, answer_id, f"savol {qid}"))
            qid += 1
    return qs


# --- Corpus cleaning ---------------------------------------------------------------------


def test_build_corpus_applies_cleaning_rules():
    raw = RawData(
        answers=pd.DataFrame(
            {
                "id": [1, 2, 3],
                "category_id": [1, 1, 2],
                "topic_id": [1, 2, 3],
                "answer": ["Javob bir", "Javob ikki", "Eski javob"],
                "status": ["active", "active", "inactive"],
            }
        ),
        questions=pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "answer_id": [1, 1, 2, 3, 99],
                "question": ["a", "b", "c", "d", "e"],
                "is_active": [True, False, "true", True, True],
            }
        ),
    )
    corpus = build_corpus(raw)
    assert set(corpus.answers) == {1, 2}
    assert [q.id for q in corpus.questions] == [1, 3]  # inactive, inactive-answer, orphan dropped


# --- Split -------------------------------------------------------------------------------


def test_split_sizes_and_no_leakage():
    qs = make_questions({1: 6, 2: 3, 3: 2, 4: 1})
    split = split_questions(qs, seed=42)
    assert len(split.test) == 3  # answers 1, 2, 3
    assert len(split.validation) == 2  # answers 1, 2 (>= 3 questions)
    assert len(split.index) == len(qs) - 5
    ids = [q.id for q in split.index + split.validation + split.test]
    assert sorted(ids) == sorted(q.id for q in qs)  # every question exactly once
    assert {q.answer_id for q in split.index} == {1, 2, 3, 4}  # every answer stays searchable


def test_split_is_deterministic():
    qs = make_questions({1: 6, 2: 6, 3: 6})
    assert split_questions(qs, 42) == split_questions(list(reversed(qs)), 42)
    assert split_questions(qs, 42) != split_questions(qs, 7)


# --- Decision + metrics ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scores", "margin", "expected"),
    [
        ([0.9, 0.1], 0.0, "answer"),
        ([0.9, 0.85], 0.1, "suggest"),  # too close to the runner-up
        ([0.4, 0.1], 0.0, "suggest"),
        ([0.1], 0.0, "not_found"),
        ([], 0.0, "not_found"),
    ],
)
def test_decide_status(scores, margin, expected):
    candidates = [Candidate(i, s) for i, s in enumerate(scores)]
    assert decide_status(candidates, 0.5, 0.2, margin) == expected


def record(gold, ranked_ids, status="answer"):
    ranked = [(a, 1.0 - i / 10) for i, a in enumerate(ranked_ids)]
    return QueryRecord(query="q", gold=gold, status=status, ranked=ranked, latency_ms=1.0)


def test_ranking_metrics():
    records = [record(1, [1, 2, 3]), record(2, [1, 2, 3]), record(3, [4, 5, 6])]
    assert top_k_accuracy(records, 1) == pytest.approx(1 / 3)
    assert top_k_accuracy(records, 3) == pytest.approx(2 / 3)
    assert mrr(records) == pytest.approx((1 + 0.5 + 0) / 3)


def test_summary_decision_metrics():
    records = [
        record(1, [1, 2], "answer"),
        record(2, [1, 2], "answer"),
        record(3, [3, 1], "suggest"),
        record(4, [5, 6], "not_found"),
    ]
    m = summarize(records)
    assert m["coverage"] == 0.5
    assert m["precision_at_answer"] == 0.5
    assert m["suggest_rate"] == 0.25
    assert m["suggest_hit_at3"] == 1.0
    assert m["not_found_rate"] == 0.25


def test_worst_failures_puts_confident_answers_first():
    confident = record(1, [9], "answer")
    hedged = record(1, [8], "suggest")
    hedged.ranked = [(8, 0.99)]
    assert worst_failures([hedged, record(1, [1]), confident]) == [confident, hedged]


def test_law_injection_uses_absent_laws():
    qs = make_questions({1: 2})
    queries = law_injection_queries(qs, known_laws={f"pq_{n}" for n in range(1000, 9000)}, seed=1)
    assert all("PQ-" not in q.text or int(q.text.split("PQ-")[1][:4]) >= 9000 for q in queries)
    assert [q.gold for q in queries] == [1, 1]


# --- End to end --------------------------------------------------------------------------


def fixture_settings(tmp_path) -> Settings:
    """Five answers with four DXM questions each, plus every auxiliary eval file."""
    answers = pd.DataFrame(
        {
            "id": range(1, 6),
            "category_id": [1] * 5,
            "topic_id": [1] * 5,
            "answer": [f"Javob {i}, DXM orqali" for i in range(1, 6)],
            "status": ["active"] * 5,
        }
    )
    questions = make_questions({i: 4 for i in range(1, 6)})
    pd.DataFrame(
        {
            "id": [q.id for q in questions],
            "answer_id": [q.answer_id for q in questions],
            "question": [f"{q.text} DXMga qanday murojaat qilinadi" for q in questions],
            "is_active": [True] * len(questions),
        }
    ).to_csv(tmp_path / "questions.csv", index=False)
    answers.to_csv(tmp_path / "answers.csv", index=False)
    (tmp_path / "ood.txt").write_text("# comment\nOb-havo qanday?\n", encoding="utf-8")
    (tmp_path / "gazetteer.yaml").write_text(
        "- {id: dxm, type: ORG, aliases: [DXM]}\n- {id: iiv, type: ORG, aliases: [IIV]}\n",
        encoding="utf-8",
    )
    pd.DataFrame({"query": ["javob ikki"], "answer_id": [2], "kind": ["paraphrase"]}).to_csv(
        tmp_path / "realistic.csv", index=False
    )
    pd.DataFrame({"query": ["javob uch"], "answer_id": [3], "kind": ["keyword"]}).to_csv(
        tmp_path / "realistic_dev.csv", index=False
    )
    return Settings(
        _env_file=None,
        answers_path=tmp_path / "answers.csv",
        questions_path=tmp_path / "questions.csv",
        ood_questions_path=tmp_path / "ood.txt",
        gazetteer_path=tmp_path / "gazetteer.yaml",
        eval_realistic_path=tmp_path / "realistic.csv",
        eval_realistic_dev_path=tmp_path / "realistic_dev.csv",
        synonyms_path=tmp_path / "no_synonyms.yaml",
        use_static_emb=False,
        reports_dir=tmp_path / "reports",
    )


def test_evaluate_end_to_end(tmp_path):
    settings = fixture_settings(tmp_path)
    ctx, reports = evaluate(settings, ["random"])
    results = write_reports(ctx, reports, reports[0], settings.reports_dir)

    m = results["systems"]["random"]
    assert results["dataset"] == {
        "answers": 5,
        "questions": 20,
        "index": 10,
        "validation": 5,
        "test": 5,
        "realistic_dev": 1,
        "realistic": 1,
    }
    assert m["test"]["n"] == 5 and m["ood"]["n"] == 1 and m["realistic"]["n"] == 1
    assert m["realistic_dev"]["n"] == 1 and "keyword" in m["realistic_dev_by_kind"]
    assert m["test_entity"]["n"] == 5  # every question mentions DXM
    assert m["org_swap"]["n"] == 5  # DXM → IIV
    assert m["law_injection"]["n"] == 5
    saved = json.loads((settings.reports_dir / "eval_results.json").read_text(encoding="utf-8"))
    assert saved["primary"] == "random"
    assert "worst test failures" in (settings.reports_dir / "eval_results.md").read_text()
    assert (settings.reports_dir / "eval_predictions.jsonl").stat().st_size > 0


def test_tune_reranker_uses_decision_sets_only(tmp_path):
    ctx = build_context(fixture_settings(tmp_path))
    baseline, rows = tune_reranker(ctx, {"rrf": [1.0], "fuzzy": [0.0, 1.0]})
    assert set(baseline) == {"validation", "realistic_dev"}
    assert rows and all(set(r) == {"weights", "validation", "realistic_dev"} for r in rows)
    keys = [(-r["realistic_dev"]["top1"], -r["realistic_dev"]["mrr10"]) for r in rows]
    assert keys == sorted(keys)
