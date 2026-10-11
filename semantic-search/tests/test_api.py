import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import main as api
from app.api.main import create_app
from app.core.config import Settings

ANSWERS = {
    1: "ID-karta Davlat xizmatlari markazi orqali almashtiriladi.",
    2: "Haydovchilik guvohnomasi imtihondan so'ng beriladi.",
    3: "Soliq qarzi soliq.uz shaxsiy kabinetida ko'rinadi. PQ-4567 qaroriga asosan.",
}
QUESTIONS = [
    (1, 1, "ID-kartani qanday almashtirish mumkin?"),
    (2, 1, "Pasportni almashtirish tartibi qanday?"),
    (3, 2, "Haydovchilik guvohnomasini qanday olaman?"),
    (4, 2, "Haydovchilik guvohnomasi uchun imtihon qanday?"),
    (5, 3, "Soliq qarzimni qanday tekshiraman?"),
    (6, 3, "Soliqdan qarzim bormi?"),
]


@pytest.fixture
def settings(tmp_path):
    pd.DataFrame(
        {
            "id": list(ANSWERS),
            "category_id": [1, 2, 3],
            "topic_id": [1, 2, 3],
            "answer": list(ANSWERS.values()),
            "status": ["active"] * 3,
        }
    ).to_csv(tmp_path / "answers.csv", index=False)
    pd.DataFrame(QUESTIONS, columns=["id", "answer_id", "question"]).assign(is_active=True).to_csv(
        tmp_path / "questions.csv", index=False
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "eval_results.json").write_text(
        json.dumps({"generated_at": "now", "systems": {"engine": {"test": {"top1": 0.99}}}})
    )
    return Settings(
        _env_file=None,
        answers_path=tmp_path / "answers.csv",
        questions_path=tmp_path / "questions.csv",
        gazetteer_path=tmp_path / "none.yaml",
        synonyms_path=tmp_path / "none.yaml",
        use_static_emb=False,
        logs_dir=tmp_path / "logs",
        reports_dir=reports,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def ask(client, question, **kwargs):
    return client.post("/ask", json={"question": question, **kwargs})


def test_health(client):
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "answers": 3,
        "questions": 6,
        "build_seconds": body["build_seconds"],
        "data_source": "file",
        "static_embeddings": False,
    }


def test_ask_answer(client):
    r = ask(client, "Soliq qarzimni qanday tekshiraman?")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answer" and body["answer_id"] == 3
    assert body["answer"] == ANSWERS[3]
    top = body["candidates"][0]
    assert top["answer_id"] == 3 and top["question"] == "Soliq qarzimni qanday tekshiraman?"
    assert top["preview"].startswith("Soliq qarzi")
    assert body["debug"] is None and body["latency_ms"] > 0


def test_ask_cyrillic(client):
    body = ask(client, "Солиқ қарзимни қандай текшираман?").json()
    assert body["answer_id"] == 3


def test_ask_unknown_law_is_not_found(client):
    body = ask(client, "PQ-9999 qarori bo'yicha soliq qarzi").json()
    assert body["status"] == "not_found" and body["reason"] == "unknown_law"
    assert body["answer"] is None
    assert body["entities"][0]["token"] == "LAW_pq_9999"


def test_ask_gibberish_is_not_found(client):
    body = ask(client, "zzzz qqqq").json()
    assert body["status"] == "not_found" and body["candidates"] == []


def test_ask_suggest_has_candidates_but_no_answer(settings):
    strict = settings.model_copy(update={"answer_threshold": 0.999, "suggest_threshold": 0.0})
    with TestClient(create_app(strict)) as c:
        body = ask(c, "Soliq qarzimni qanday tekshiraman?").json()
    assert body["status"] == "suggest"
    assert body["answer"] is None and body["answer_id"] is None
    assert body["candidates"] and body["candidates"][0]["question"]


def test_ask_debug_and_top_k(client):
    body = ask(client, "soliq qarzi", debug=True, top_k=2).json()
    assert len(body["candidates"]) <= 2
    assert "candidates" in body["debug"] and "processed_query" in body["debug"]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 501},
        {"question": "soliq", "top_k": 0},
        {"question": "soliq", "top_k": 11},
        {},
    ],
)
def test_ask_validation_errors(client, payload):
    r = client.post("/ask", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"][0]["msg"]


def test_blank_question_message(client):
    detail = client.post("/ask", json={"question": "  "}).json()["detail"][0]
    assert "must not be empty" in detail["msg"]


def test_get_answer(client):
    body = client.get("/answers/2").json()
    assert body["answer"] == ANSWERS[2]
    assert body["questions"] == [q for _, a, q in QUESTIONS if a == 2]
    missing = client.get("/answers/999")
    assert missing.status_code == 404 and "999" in missing.json()["detail"]


def test_feedback(client, settings):
    r = client.post("/feedback", json={"query": "soliq", "answer_id": 3, "helpful": True})
    assert r.status_code == 201 and r.json() == {"id": 1}
    assert client.post("/feedback", json={"query": "x", "helpful": False}).json() == {"id": 2}
    lines = (settings.logs_dir / "feedback.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["answer_id"] == 3
    unknown = client.post("/feedback", json={"query": "x", "answer_id": 999, "helpful": True})
    assert unknown.status_code == 404


def test_stats_include_eval_and_usage(client):
    ask(client, "Soliq qarzimni qanday tekshiraman?")
    ask(client, "zzzz qqqq")
    body = client.get("/stats").json()
    assert body["data"] == {"answers": 3, "questions": 6}
    assert body["config"]["answer_threshold"] == 0.68
    assert body["eval"]["test"]["top1"] == 0.99
    usage = body["usage"]
    assert usage["total_queries"] == 2
    assert usage["status_rates"] == {"answer": 0.5, "suggest": 0.0, "not_found": 0.5}


def test_reload_swaps_engine(client):
    before = client.app.state.engine_state
    r = client.post("/reload")
    assert r.status_code == 200 and r.json()["answers"] == 3
    assert client.app.state.engine_state is not before
    assert ask(client, "Soliq qarzimni qanday tekshiraman?").json()["answer_id"] == 3


def test_failed_reload_keeps_old_engine(client, monkeypatch):
    before = client.app.state.engine_state

    def broken(settings):
        raise RuntimeError("data file missing")

    monkeypatch.setattr(api, "build_state", broken)
    r = client.post("/reload")
    assert r.status_code == 500 and "data file missing" in r.json()["detail"]
    assert client.app.state.engine_state is before
    assert ask(client, "soliq qarzi").status_code == 200


def test_query_log_failure_does_not_break_ask(settings):
    class BrokenRepo:
        def log_query(self, entry):
            raise OSError("disk full")

        def save_feedback(self, entry):
            return 1

        def usage_stats(self):
            return {}

    with TestClient(create_app(settings, repo=BrokenRepo())) as c:
        assert ask(c, "soliq qarzi").status_code == 200
