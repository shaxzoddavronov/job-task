"""Query results and the answer / suggest / not_found decision.

QAEngine (Phase 4) and the evaluation baselines both return QAResult, so the evaluation code
does not care which system produced it.
"""

from dataclasses import dataclass, field
from typing import Literal, Protocol

from app.core.entities import Entity

Status = Literal["answer", "suggest", "not_found"]


@dataclass(frozen=True)
class Candidate:
    answer_id: int
    score: float


@dataclass(frozen=True)
class QAResult:
    status: Status
    candidates: list[Candidate]  # best first
    entities: list[Entity] = field(default_factory=list)
    debug: dict | None = None

    @property
    def top(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def confidence(self) -> float:
        return self.candidates[0].score if self.candidates else 0.0


class QASystem(Protocol):
    def ask(self, query: str, k: int = 10) -> QAResult: ...


def decide_status(
    candidates: list[Candidate],
    answer_threshold: float,
    suggest_threshold: float,
    min_margin: float = 0.0,
) -> Status:
    """Answer only when the top score is high and clearly ahead of the runner-up."""
    if not candidates or candidates[0].score < suggest_threshold:
        return "not_found"
    margin = candidates[0].score - (candidates[1].score if len(candidates) > 1 else 0.0)
    if candidates[0].score >= answer_threshold and margin >= min_margin:
        return "answer"
    return "suggest"
