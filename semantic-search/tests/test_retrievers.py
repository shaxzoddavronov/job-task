import numpy as np
import pytest

from app.core.retrievers.base import top_k
from app.core.retrievers.bm25 import BM25Retriever
from app.core.retrievers.tfidf_char import CharTfidfRetriever

DOCS = [
    "pasport id karta qanday almashtiriladi",
    "haydovchilik guvohnomasi qanday olinadi",
    "soliq qarzini qanday tekshirish mumkin",
    "LAW_pq_4567 qarori nima haqida",
    "LAW_pq_4576 qarori nima haqida",
]


@pytest.fixture(params=["tfidf", "bm25"])
def retriever(request):
    r = (
        CharTfidfRetriever((3, 5), entity_weight=1.0)
        if request.param == "tfidf"
        else BM25Retriever()
    )
    r.fit(DOCS)
    return r


def test_obvious_document_ranks_first(retriever):
    assert retriever.search("haydovchilik guvohnomasi olish", k=3)[0][0] == 1


def test_entity_number_decides(retriever):
    hits = retriever.search("LAW_pq_4567 qarori", k=5)
    assert hits[0][0] == 3
    scores = dict(hits)
    assert scores[3] > scores.get(4, 0.0)


def test_k_larger_than_corpus_is_safe(retriever):
    hits = retriever.search("qanday", k=100)
    assert 0 < len(hits) <= len(DOCS)
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


def test_empty_query_returns_nothing(retriever):
    assert retriever.search("", k=5) == []


def test_tfidf_typo_tolerance():
    r = CharTfidfRetriever((3, 5))
    r.fit(DOCS)
    assert r.search("haydovchlik guvohnomas", k=1)[0][0] == 1


def test_tfidf_without_entities_in_corpus():
    r = CharTfidfRetriever()
    r.fit(["birinchi hujjat", "ikkinchi hujjat"])
    assert r.search("LAW_pq_1 birinchi", k=1)[0][0] == 0


def test_top_k_drops_zero_scores_and_sorts():
    assert top_k(np.array([0.0, 0.5, 0.2, 0.0]), 3) == [(1, 0.5), (2, 0.2)]
    assert top_k(np.array([]), 3) == []
