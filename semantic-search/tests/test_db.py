"""PostgreSQL tests. Run only when TEST_DATABASE_URL is set (`make test-db`); they create and
reset their own database so the main data is never touched."""

import argparse
import os
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.db

URL = os.environ.get("TEST_DATABASE_URL")
if not URL:
    pytest.skip("TEST_DATABASE_URL not set", allow_module_level=True)

from alembic import command  # noqa: E402
from alembic.autogenerate import compare_metadata  # noqa: E402
from alembic.config import Config  # noqa: E402
from alembic.migration import MigrationContext  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402

from app.core.data import PostgresSource, RawData, build_corpus  # noqa: E402
from app.db.models import Base  # noqa: E402
from app.db.repository import Feedback, PostgresRepository, QueryLog  # noqa: E402
from app.db.session import get_engine  # noqa: E402
from scripts.import_xlsx import import_raw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def raw_data() -> RawData:
    answers = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "category_id": [1, 1, 2],
            "topic_id": [1, 2, 3],
            "answer": ["Javob bir.", "Javob ikki.", "Eski javob."],
            "status": ["active", "active", "inactive"],
            "created_at": ["2026-08-04 00:00:00"] * 3,
            "updated_at": ["2026-08-04 00:00:00"] * 3,
        }
    )
    questions = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "answer_id": [1, 1, 2, 3, 99],  # 99 is an orphan
            "question": ["Birinchi savol?", "Ikkinchi savol?", "Uchinchi?", "Eski?", "Yetim?"],
            "is_active": ["true", "true", "false", "true", "true"],
            "created_at": ["2026-08-04 00:00:00"] * 5,
            "updated_at": ["2026-08-04 00:00:00"] * 5,
        }
    )
    return RawData(answers, questions)


@pytest.fixture(scope="module")
def db_url():
    url = make_url(URL)
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": url.database}
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    admin.dispose()
    return URL


@pytest.fixture
def migrated(db_url):
    """Fresh schema at alembic head for every test."""
    engine = get_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.cmd_opts = argparse.Namespace(x=[f"url={db_url}"])
    command.upgrade(cfg, "head")
    return db_url


def test_migration_matches_models(migrated):
    with get_engine(migrated).connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []


def test_import_counts_orphans_and_idempotency(migrated):
    first = import_raw(raw_data(), migrated)
    assert (first.answers_inserted, first.questions_inserted, first.orphan_questions) == (3, 4, 1)
    second = import_raw(raw_data(), migrated)
    assert (second.answers_inserted, second.answers_updated) == (0, 0)
    assert (second.questions_inserted, second.questions_updated) == (0, 0)
    assert import_raw(raw_data(), migrated, if_empty=True).skipped


def test_import_updates_changed_rows(migrated):
    import_raw(raw_data(), migrated)
    changed = raw_data()
    changed.answers.loc[0, "answer"] = "Yangilangan javob."
    report = import_raw(changed, migrated)
    assert (report.answers_inserted, report.answers_updated) == (0, 1)


def test_postgres_source_matches_file_source(migrated):
    raw = raw_data()
    import_raw(raw, migrated)
    from_db = build_corpus(PostgresSource(migrated).load_raw())
    imported = RawData(raw.answers, raw.questions[raw.questions["answer_id"] != 99])
    assert from_db == build_corpus(imported)
    assert set(from_db.answers) == {1, 2} and [q.id for q in from_db.questions] == [1, 2]


def test_postgres_repository(migrated):
    import_raw(raw_data(), migrated)
    repo = PostgresRepository(migrated)
    for status in ("answer", "answer", "not_found"):
        repo.log_query(QueryLog("Savol?", "savol", status, 1, 0.9, 5.0, "bot"))
    assert repo.save_feedback(Feedback("Savol?", 1, True, "bot")) == 1
    assert repo.save_feedback(Feedback("Savol?", None, False, "api")) == 2
    stats = repo.usage_stats()
    assert stats["total_queries"] == 3
    assert stats["status_rates"] == pytest.approx(
        {"answer": 2 / 3, "suggest": 0, "not_found": 1 / 3}
    )
    assert stats["avg_latency_ms"] == pytest.approx(5.0)
    assert stats["top_questions"] == [{"question": "savol", "count": 3}]
    assert stats["feedback"] == {"total": 2, "helpful": 1}
