import pandas as pd
import pytest

from app.core.data import ANSWER_COLUMNS, QUESTION_COLUMNS, FileSource
from scripts.audit_data import audit


@pytest.fixture
def raw(tmp_path):
    answers = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "category_id": [10, 10, 20, 20],
            "topic_id": [100, 101, 200, 201],
            "answer": [
                "PQ-4567 qaroriga asosan Adliya vazirligi ariza qabul qiladi.",
                "Mehnat kodeksining 15-moddasi bo'yicha ta'til beriladi.",
                "Ha.",
                "Паспорт олиш учун ДХМ га мурожаат қилинг.",
            ],
            "status": [1, 1, 1, 0],
        }
    )
    questions = pd.DataFrame(
        {
            "id": range(1, 10),
            "answer_id": [1, 1, 1, 2, 2, 3, 4, 99, 2],
            "question": [
                "PQ-4567 qarori nima haqida?",
                "pq 4567 qaror haqida ma’lumot",
                "Qaror haqida",
                "Ta‘til qancha beriladi?",
                "Ta'til qancha beriladi",
                "Ta'til qancha beriladi?",  # conflict: same text as id 5 but answer 3
                "Паспорт қандай олинади?",
                "Yetim savol",  # orphan
                "Eski savol",  # inactive
            ],
            "is_active": [1, 1, 1, 1, 1, 1, 1, 1, 0],
        }
    )
    a_path, q_path = tmp_path / "answers.xlsx", tmp_path / "questions.xlsx"
    answers.to_excel(a_path, index=False)
    questions.to_excel(q_path, index=False)
    return FileSource(a_path, q_path).load_raw()


def test_loader_rejects_missing_columns(tmp_path):
    path = tmp_path / "answers.xlsx"
    pd.DataFrame({"id": [1]}).to_excel(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        FileSource(path, path).load_raw()


def test_loader_reads_csv(tmp_path):
    a_path, q_path = tmp_path / "answers.csv", tmp_path / "questions.csv"
    pd.DataFrame({c: [1] for c in ANSWER_COLUMNS}).to_csv(a_path, index=False)
    pd.DataFrame({c: [1] for c in QUESTION_COLUMNS}).to_csv(q_path, index=False)
    raw = FileSource(a_path, q_path).load_raw()
    assert len(raw.answers) == len(raw.questions) == 1


def test_loader_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        FileSource(tmp_path / "a.xlsx", tmp_path / "q.xlsx").load_raw()


def test_audit_finds_planted_issues(raw):
    md, summary = audit(raw)
    assert summary["counts"]["questions_active"] == 8
    assert summary["counts"]["questions_inactive"] == 1
    assert summary["integrity"]["orphan_active_questions"] == 1
    assert summary["duplicates"]["conflicting_texts"] == 1
    assert summary["questions_per_answer"]["eval_pool_ge2"] == 2
    assert summary["questions_per_answer"]["with_validation_ge3"] == 1
    assert summary["script"]["questions_with_cyrillic"] == 1
    assert summary["entities"]["law number"]["questions"] == 2
    assert summary["entities"]["questions_with_law_whose_answer_has_it"] == 2
    assert summary["short_texts"]["short_answers(<30 chars)"] == 1
    assert md[0] == "# Data audit"
