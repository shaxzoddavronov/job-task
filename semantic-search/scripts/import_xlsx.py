"""Import answers and questions (CSV or xlsx via FileSource) into PostgreSQL. Idempotent.

Usage: python -m scripts.import_xlsx [--if-empty]   (or `make import` / `make import-if-empty`)
Upserts by id, skips questions whose answer is missing (Phase 1 rule) and prints counts.
--if-empty does nothing when the answers table already has rows (first-run import in Docker).
"""

import argparse
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert

from app.core.config import get_settings
from app.core.data import FileSource, RawData, as_bool
from app.db.models import Answer, Question
from app.db.session import get_engine

BATCH = 500


@dataclass
class ImportReport:
    answers_inserted: int = 0
    answers_updated: int = 0
    questions_inserted: int = 0
    questions_updated: int = 0
    orphan_questions: int = 0
    skipped: bool = False


def _int(value) -> int | None:
    return None if pd.isna(value) else int(value)


def _ts(value):
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).to_pydatetime().replace(tzinfo=None)


def _upsert(conn, model, rows: list[dict]) -> tuple[int, int]:
    """Insert or update rows by id; returns (inserted, updated). Unchanged rows are skipped."""
    inserted = updated = 0
    columns = [c for c in rows[0] if c != "id"] if rows else []
    for start in range(0, len(rows), BATCH):
        stmt = insert(model).values(rows[start : start + BATCH])
        changed = " OR ".join(
            f"{model.__tablename__}.{c} IS DISTINCT FROM excluded.{c}" for c in columns
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={c: stmt.excluded[c] for c in columns},
            where=text(changed),
        ).returning(text("(xmax = 0) AS inserted"))
        for (was_insert,) in conn.execute(stmt):
            inserted += was_insert
            updated += not was_insert
    return inserted, updated


def import_raw(raw: RawData, database_url: str, if_empty: bool = False) -> ImportReport:
    report = ImportReport()
    engine = get_engine(database_url)
    with engine.begin() as conn:
        if if_empty and conn.execute(select(func.count()).select_from(Answer)).scalar():
            report.skipped = True
            return report
        answers = [
            {
                "id": int(r.id),
                "category_id": _int(r.category_id),
                "topic_id": _int(r.topic_id),
                "answer": str(r.answer),
                "status": None if pd.isna(r.status) else str(r.status),
                "created_at": _ts(getattr(r, "created_at", None)),
                "updated_at": _ts(getattr(r, "updated_at", None)),
            }
            for r in raw.answers.itertuples()
        ]
        report.answers_inserted, report.answers_updated = _upsert(conn, Answer, answers)

        known = set(conn.execute(select(Answer.id)).scalars())
        questions = []
        for r in raw.questions.itertuples():
            if int(r.answer_id) not in known:
                report.orphan_questions += 1
                continue
            questions.append(
                {
                    "id": int(r.id),
                    "answer_id": int(r.answer_id),
                    "question": str(r.question),
                    "is_active": bool(as_bool(r.is_active)),
                    "created_at": _ts(getattr(r, "created_at", None)),
                    "updated_at": _ts(getattr(r, "updated_at", None)),
                }
            )
        report.questions_inserted, report.questions_updated = _upsert(conn, Question, questions)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--if-empty", action="store_true", help="skip if answers already exist")
    args = parser.parse_args()
    settings = get_settings()
    raw = FileSource(settings.answers_path, settings.questions_path).load_raw()
    report = import_raw(raw, settings.database_url, if_empty=args.if_empty)
    if report.skipped:
        print("Database already has answers; nothing imported (--if-empty).")
        return
    print(
        f"answers: {report.answers_inserted} inserted, {report.answers_updated} updated · "
        f"questions: {report.questions_inserted} inserted, {report.questions_updated} updated · "
        f"orphan questions skipped: {report.orphan_questions}"
    )


if __name__ == "__main__":
    main()
