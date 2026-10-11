import threading

import numpy as np
import pytest

from app.core.config import Settings
from app.core.data import Answer, Corpus, Question
from app.core.encoders import HashingEncoder, OverlapScorer
from app.core.engine import QAEngine
from app.core.ranker import rerank
from app.core.retrievers.dense import best_per_group, encode_cached
from app.eval.systems import SYSTEMS


@pytest.fixture
def corpus():
    answers = {
        1: Answer(1, "ID-karta Davlat xizmatlari markazi orqali almashtiriladi.", 1),
        2: Answer(2, "Haydovchilik guvohnomasi imtihondan so'ng beriladi.", 2),
        3: Answer(
            3, "Soliq qarzi soliq.uz shaxsiy kabinetida ko'rinadi. PQ-4567 qaroriga asosan.", 3
        ),
    }
    questions = [
        Question(1, 1, "ID-kartani qanday almashtirish mumkin?"),
        Question(2, 1, "Pasportni almashtirish tartibi qanday?"),
        Question(3, 2, "Haydovchilik guvohnomasini qanday olaman?"),
        Question(4, 2, "Haydovchilik guvohnomasi uchun imtihon qanday?"),
        Question(5, 3, "Soliq qarzimni qanday tekshiraman?"),
        Question(6, 3, "Soliqdan qarzim bormi?"),
    ]
    return Corpus(answers, questions)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        model_backend="fake",
        gazetteer_path=tmp_path / "none.yaml",
        synonyms_path=tmp_path / "none.yaml",
        embedding_cache_dir=None,
    )


@pytest.fixture
def engine(corpus, settings):
    return QAEngine(corpus, settings)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("ID-kartani qanday almashtirish mumkin?", 1),
        ("ИД-картани қандай алмаштириш мумкин?", 1),  # Cyrillic → Latin before encoding
        ("haydovchilik guvohnomasini olaman", 2),
        ("soliq qarzimni tekshiraman", 3),
    ],
)
def test_right_answer_first(engine, query, expected):
    result = engine.ask(query)
    assert result.top.answer_id == expected
    assert 0.0 < result.confidence <= 1.0


def test_fake_backend_runs_dense_mode_on_cpu(engine):
    assert (engine.device, engine.mode) == ("cpu", "dense")
    assert engine.models["mode"] == "dense" and engine.models["reranker"] is None


def test_reranker_on_switches_mode_and_thresholds(corpus, settings):
    on = settings.model_copy(update={"reranker": "on", "answer_threshold_rerank": 0.11})
    engine = QAEngine(corpus, on)
    assert engine.mode == "rerank" and engine.thresholds[0] == 0.11
    features = engine.retrieve("soliq qarzimni tekshiraman").candidates[0].features
    assert features["rerank"] > 0


def test_injected_models_are_used(corpus, settings):
    engine = QAEngine(
        corpus,
        settings.model_copy(update={"reranker": "on"}),
        encoder=HashingEncoder(dim=128),
        scorer=OverlapScorer(),
    )
    assert engine.question_vectors.shape == (6, 128)


def test_unknown_law_is_not_found(engine):
    unknown = engine.ask("PQ-9999 qarori bo'yicha soliq qarzi")
    assert unknown.status == "not_found" and unknown.reason == "unknown_law"
    assert unknown.candidates
    assert engine.ask("PQ-4567 qarori bo'yicha soliq qarzi").reason is None


def test_entity_match_can_promote_answer_only_entity(corpus, settings):
    weights = {"dense": 0.2, "entity_match": 1.0, "entity_conflict": 3.0}
    engine = QAEngine(corpus, settings.model_copy(update={"weights_dense": weights}))
    assert engine.ask("PQ-4567 haqida").top.answer_id == 3


def test_debug_has_mode_and_features(engine):
    result = engine.ask("soliq qarzi", debug=True)
    assert result.debug["mode"] == "dense"
    top = result.debug["candidates"][str(result.top.answer_id)]
    assert set(top["features"]) >= {"rerank", "dense", "entity_match", "entity_conflict"}
    assert engine.ask("soliq qarzi").debug is None


def test_retrieve_then_rerank_matches_ask(engine):
    r = engine.retrieve("soliq qarzi")
    scored = rerank(r.candidates, engine.weights, bool(r.tagged.tokens))
    assert [a for a, _ in scored[:3]] == [c.answer_id for c in engine.ask("soliq qarzi").candidates]


def test_status_follows_thresholds(corpus, settings):
    always = QAEngine(
        corpus,
        settings.model_copy(update={"answer_threshold_dense": 0.0, "suggest_threshold_dense": 0.0}),
    )
    never = QAEngine(corpus, settings.model_copy(update={"suggest_threshold_dense": 1.01}))
    assert always.ask("soliq qarzi").status == "answer"
    assert never.ask("soliq qarzi").status == "not_found"


def test_empty_query_and_k(engine):
    assert engine.ask("   ").candidates == []
    assert len(engine.ask("qanday", k=2).candidates) <= 2


def test_concurrent_asks_are_safe(engine):
    errors = []

    def worker():
        try:
            for _ in range(20):
                assert engine.ask("soliq qarzimni tekshiraman").top.answer_id == 3
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_best_per_group():
    scores = np.array([0.1, 0.9, 0.4, 0.9, 0.2], dtype=np.float32)
    groups = np.array([0, 0, 1, 1, 2])
    best, arg = best_per_group(scores, groups, 3)
    assert best.tolist() == pytest.approx([0.9, 0.9, 0.2])
    assert arg.tolist() == [1, 3, 4]


def test_embedding_cache_reuses_vectors(tmp_path):
    class CountingEncoder(HashingEncoder):
        cache_id = "counting@1"
        calls = 0

        def encode(self, texts):
            CountingEncoder.calls += 1
            return super().encode(texts)

    enc = CountingEncoder()
    first = encode_cached(enc, ["a b", "c d"], tmp_path)
    second = encode_cached(enc, ["a b", "c d"], tmp_path)
    assert CountingEncoder.calls == 1 and np.array_equal(first, second)
    encode_cached(enc, ["a b", "changed"], tmp_path)  # different texts → new key
    assert CountingEncoder.calls == 2


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_every_registered_system_builds_and_answers(corpus, settings, name):
    result = SYSTEMS[name](corpus, settings).ask("soliq qarzi", k=3)
    assert result.status in {"answer", "suggest", "not_found"}
    assert len(result.candidates) <= 3
