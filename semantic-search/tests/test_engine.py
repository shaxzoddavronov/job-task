import pytest

from app.core.config import Settings
from app.core.data import Answer, Corpus, Question
from app.core.engine import QAEngine
from app.core.ranker import rerank
from app.eval.systems import SYSTEMS


@pytest.fixture
def corpus():
    answers = {
        1: Answer(1, "ID-karta Davlat xizmatlari markazi orqali almashtiriladi."),
        2: Answer(2, "Haydovchilik guvohnomasi imtihondan so'ng beriladi."),
        3: Answer(3, "Soliq qarzi soliq.uz shaxsiy kabinetida ko'rinadi. PQ-4567 qaroriga asosan."),
    }
    questions = [
        Question(1, 1, "ID-kartani qanday almashtirish mumkin?"),
        Question(2, 1, "Pasportni almashtirish tartibi qanday?"),
        Question(3, 2, "Haydovchilik guvohnomasini qanday olaman?"),
        Question(4, 2, "Prava olish uchun nima kerak?"),
        Question(5, 3, "Soliq qarzimni qanday tekshiraman?"),
        Question(6, 3, "Soliqdan qarzim bormi?"),
    ]
    return Corpus(answers, questions)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        gazetteer_path=tmp_path / "none.yaml",
        synonyms_path=tmp_path / "none.yaml",
        use_bm25=True,
        use_static_emb=False,
    )


@pytest.fixture
def engine(corpus, settings):
    return QAEngine(corpus, settings)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("ID karta almashtirish", 1),
        ("ИД-картани қандай алмаштираман?", 1),  # Cyrillic
        ("haydovchilk guvohnomasi olsih", 2),  # typos
        ("soliq qarzi tekshirish", 3),
    ],
)
def test_right_answer_first(engine, query, expected):
    result = engine.ask(query)
    assert result.top.answer_id == expected
    assert 0.0 < result.confidence <= 1.0


def test_answer_index_finds_entity_only_in_answers(corpus, settings):
    # PQ-4567 appears only in answer 3's text, never in its questions.
    engine = QAEngine(corpus, settings.model_copy(update={"answer_index_weight": 1.0}))
    assert engine.ask("PQ-4567 haqida").top.answer_id == 3


def test_reranker_promotes_entity_only_in_answers(corpus, settings):
    # Mechanism test with explicit weights: the answer index alone is only a tie-breaker, and
    # entity_match (entities from questions or answer) lifts answer 3.
    weights = {"rerank_w_rrf": 0.5, "rerank_w_entity_match": 1.0}
    engine = QAEngine(corpus, settings.model_copy(update=weights))
    assert engine.ask("PQ-4567 haqida").top.answer_id == 3


def test_unknown_law_is_not_found(engine):
    unknown = engine.ask("PQ-9999 qarori bo'yicha soliq qarzi")
    assert unknown.status == "not_found" and unknown.reason == "unknown_law"
    assert unknown.candidates  # still returned for debugging
    known = engine.ask("PQ-4567 qarori bo'yicha soliq qarzi")
    assert known.reason is None


def test_debug_has_features_and_channels(engine):
    result = engine.ask("soliq qarzi", debug=True)
    assert result.debug["processed_query"] == "soliq qarzi"
    top = result.debug["candidates"][str(result.top.answer_id)]
    assert set(top["features"]) >= {"rrf", "sim_q", "fuzzy", "entity_match", "entity_conflict"}
    assert set(top["channels"]) <= {
        "tfidf:questions",
        "bm25:questions",
        "tfidf:answers",
        "bm25:answers",
    }
    assert engine.ask("soliq qarzi").debug is None


def test_retrieve_then_rerank_matches_ask(engine):
    r = engine.retrieve("soliq qarzi")
    scored = rerank(r.candidates, engine.weights, bool(r.tagged.tokens))
    assert [a for a, _ in scored[:3]] == [c.answer_id for c in engine.ask("soliq qarzi").candidates]


def test_synonyms_expand_retrieval_query(corpus, settings, tmp_path):
    path = tmp_path / "syn.yaml"
    path.write_text("- {canonical: haydovchilik guvohnomasi, variants: [voditelskiy]}\n")
    engine = QAEngine(corpus, settings.model_copy(update={"synonyms_path": path}))
    result = engine.ask("voditelskiy olish", debug=True)
    assert result.debug["retrieval_query"] == "voditelskiy olish haydovchilik guvohnomasi"
    assert result.top.answer_id == 2


def test_status_follows_thresholds(corpus, settings):
    always = QAEngine(corpus, settings.model_copy(update={"answer_threshold": 0.0}))
    never = QAEngine(corpus, settings.model_copy(update={"suggest_threshold": 1.01}))
    assert always.ask("soliq qarzi").status == "answer"
    assert never.ask("soliq qarzi").status == "not_found"


def test_k_limits_candidates_and_latency_is_recorded(engine):
    result = engine.ask("qanday", k=2)
    assert len(result.candidates) <= 2
    assert result.latency_ms > 0


def test_no_retriever_enabled_is_an_error(corpus, settings):
    with pytest.raises(ValueError, match="No retriever"):
        QAEngine(corpus, settings.model_copy(update={"use_tfidf": False, "use_bm25": False}))


_NEEDS_MODEL = {"7a +model2vec pretrained", "7b +model2vec distilled bge-m3", "engine"}


@pytest.mark.parametrize("name", [n for n in SYSTEMS if n not in _NEEDS_MODEL])
def test_every_registered_system_builds_and_answers(corpus, settings, name):
    result = SYSTEMS[name](corpus, settings).ask("soliq qarzi", k=3)
    assert result.status in {"answer", "suggest", "not_found"}
    assert len(result.candidates) <= 3
