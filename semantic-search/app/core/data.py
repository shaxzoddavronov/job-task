"""Data sources. FileSource reads the raw CSV/xlsx files; cleaning comes after the audit."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
