"""Data loading and cleaning.

FileSource reads the raw CSV/xlsx files. build_corpus applies the approved cleaning rules
(Phase 1: keep active questions and answers, drop orphans, drop nothing else) and returns the
Corpus the engine and the evaluation work on.
"""

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ANSWER_COLUMNS = ("id", "category_id", "topic_id", "answer", "status")
QUESTION_COLUMNS = ("id", "answer_id", "question", "is_active")


@dataclass(frozen=True)
class RawData:
    answers: pd.DataFrame
    questions: pd.DataFrame


def _read_table(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Put the data file in data/ (see README).")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8")
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported data file type: {path.name} (use .csv or .xlsx)")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}; found {list(df.columns)}")
    return df


class FileSource:
    """Loads answers and questions from CSV or xlsx files, unmodified."""

    def __init__(self, answers_path: Path, questions_path: Path) -> None:
        self.answers_path = answers_path
        self.questions_path = questions_path

    def load_raw(self) -> RawData:
        return RawData(
            answers=_read_table(self.answers_path, ANSWER_COLUMNS),
            questions=_read_table(self.questions_path, QUESTION_COLUMNS),
        )


def as_bool(value) -> bool | None:
    """Parse the flag values seen in exports (True, 1, "true", "active", ...). None if unknown."""
    if pd.isna(value):
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y", "ha", "active"}:
        return True
    if text in {"0", "0.0", "false", "f", "no", "n", "yo'q", "inactive"}:
        return False
    return None


@dataclass(frozen=True)
class Answer:
    id: int
    text: str
    category_id: int | None = None
    topic_id: int | None = None


@dataclass(frozen=True)
class Question:
    id: int
    answer_id: int
    text: str


@dataclass(frozen=True)
class Corpus:
    """Answers plus the questions that are searchable for them."""

    answers: dict[int, Answer]
    questions: list[Question] = field(default_factory=list)

    def with_questions(self, questions: list[Question]) -> "Corpus":
        return replace(self, questions=list(questions))


def _optional_int(value) -> int | None:
    return None if pd.isna(value) else int(value)


def build_corpus(raw: RawData) -> Corpus:
    """Apply the approved cleaning rules and convert rows to typed records."""
    answers = raw.answers[raw.answers["status"].map(as_bool) != False]  # noqa: E712
    by_id = {
        int(r.id): Answer(
            int(r.id),
            str(r.answer).strip(),
            _optional_int(r.category_id),
            _optional_int(r.topic_id),
        )
        for r in answers.itertuples()
        if isinstance(r.answer, str) and r.answer.strip()
    }
    questions, orphans = [], 0
    for r in raw.questions.itertuples():
        if as_bool(r.is_active) is not True or not str(r.question).strip():
            continue
        if int(r.answer_id) not in by_id:
            orphans += 1
            continue
        questions.append(Question(int(r.id), int(r.answer_id), str(r.question).strip()))
    if orphans:
        log.warning("Dropped %d active questions whose answer is missing or inactive", orphans)
    return Corpus(by_id, questions)
