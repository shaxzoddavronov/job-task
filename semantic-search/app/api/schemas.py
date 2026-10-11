"""Request / response models. Examples are in Uzbek so /docs works out of the box."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import get_settings

Status = Literal["answer", "suggest", "not_found"]
Source = Literal["api", "bot"]
_MAX_CHARS = get_settings().max_question_chars


class AskRequest(BaseModel):
    question: str = Field(max_length=_MAX_CHARS, description="Savol (lotin yoki kirill)")
    top_k: int = Field(3, ge=1, le=10, description="Nechta nomzod qaytarilsin")
    debug: bool = Field(False, description="Har bir retriever va belgi bo'yicha tafsilotlar")
    source: Source = "api"

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"question": "ID-karta qancha muddatga beriladi?", "top_k": 3},
                {"question": "prava olish uchun nima kerak", "top_k": 3, "debug": True},
                {"question": "Паспортни қандай алмаштираман?"},
                {"question": "PQ-9999 qarori nima haqida?"},
                {"question": "Bugun ob-havo qanday?"},
            ]
        }
    )

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class EntityOut(BaseModel):
    type: str
    value: str
    token: str
    surface: str


class CandidateOut(BaseModel):
    answer_id: int
    score: float = Field(description="Re-ranker confidence in [0, 1]")
    question: str = Field(description="A representative question for this answer")
    preview: str = Field(description="Start of the answer text")


class AskResponse(BaseModel):
    status: Status
    answer_id: int | None = Field(None, description="Set only when status is 'answer'")
    answer: str | None = Field(None, description="Set only when status is 'answer'")
    confidence: float
    reason: str | None = Field(None, description="e.g. 'unknown_law' for a law not in the data")
    candidates: list[CandidateOut]
    entities: list[EntityOut]
    debug: dict | None = None
    latency_ms: float

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "answer",
                    "answer_id": 5,
                    "answer": "Bir yoshgacha bolaga — 2 yil, 1 yoshdan 16 yoshgacha — 5 yil.",
                    "confidence": 0.86,
                    "reason": None,
                    "candidates": [
                        {
                            "answer_id": 5,
                            "score": 0.86,
                            "question": "ID-karta qancha muddatga beriladi?",
                            "preview": "Bir yoshgacha bo'lgan bolaga — 2 yil...",
                        }
                    ],
                    "entities": [
                        {
                            "type": "TERM",
                            "value": "id_karta",
                            "token": "TERM_id_karta",
                            "surface": "ID-karta",
                        }
                    ],
                    "debug": None,
                    "latency_ms": 5.2,
                }
            ]
        }
    )


class AnswerOut(BaseModel):
    answer_id: int
    answer: str
    category_id: int | None
    topic_id: int | None
    questions: list[str]


class HealthOut(BaseModel):
    status: Literal["ok"]
    answers: int
    questions: int
    build_seconds: float
    data_source: str
    device: str = Field(description="cuda or cpu")
    mode: str = Field(description="rerank (bi-encoder + cross-encoder) or dense (bi-encoder only)")


class StatsOut(BaseModel):
    data: dict
    config: dict
    eval: dict | None = Field(None, description="Last `make eval` metrics, if available")
    usage: dict


class FeedbackIn(BaseModel):
    query: str = Field(min_length=1, max_length=_MAX_CHARS)
    answer_id: int | None = None
    helpful: bool
    source: Source = "api"

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "ID-karta qancha muddatga beriladi?", "answer_id": 5, "helpful": True}
            ]
        }
    )


class FeedbackOut(BaseModel):
    id: int


class ReloadOut(BaseModel):
    status: Literal["reloaded"]
    answers: int
    questions: int
    build_seconds: float
