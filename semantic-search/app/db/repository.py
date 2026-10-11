"""Storage for query logs and feedback (never used for search).

`Repository` is the interface the API depends on. `PostgresRepository` writes to the
`query_logs` / `feedback` tables (DATA_SOURCE=postgres); `JsonlRepository` writes JSON lines to
data/logs/ for file-only runs. `build_repository` picks one from the settings.
"""

import json
import threading
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.core.config import Settings


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class QueryLog:
    query: str
    normalized_query: str
    status: str
    answer_id: int | None
    confidence: float
    latency_ms: float
    source: str
    created_at: str = field(default_factory=_now)


@dataclass(frozen=True)
class Feedback:
    query: str
    answer_id: int | None
    helpful: bool
    source: str
    created_at: str = field(default_factory=_now)


class Repository(Protocol):
    def log_query(self, entry: QueryLog) -> None: ...

    def save_feedback(self, entry: Feedback) -> int: ...

    def usage_stats(self) -> dict: ...


class JsonlRepository:
    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.queries_path = self.logs_dir / "query_logs.jsonl"
        self.feedback_path = self.logs_dir / "feedback.jsonl"
        self._lock = threading.Lock()

    def _append(self, path: Path, record: dict) -> None:
        with self._lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _read(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def log_query(self, entry: QueryLog) -> None:
        self._append(self.queries_path, asdict(entry))

    def save_feedback(self, entry: Feedback) -> int:
        with self._lock:
            new_id = len(self._read(self.feedback_path)) + 1
            with self.feedback_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": new_id, **asdict(entry)}, ensure_ascii=False) + "\n")
        return new_id

    def usage_stats(self) -> dict:
        logs = self._read(self.queries_path)
        feedback = self._read(self.feedback_path)
        n = len(logs)
        statuses = Counter(r["status"] for r in logs)
        return {
            "total_queries": n,
            "status_rates": {s: statuses.get(s, 0) / n for s in ("answer", "suggest", "not_found")}
            if n
            else {},
            "avg_latency_ms": sum(r["latency_ms"] for r in logs) / n if n else None,
            "top_questions": [
                {"question": q, "count": c}
                for q, c in Counter(r["normalized_query"] for r in logs).most_common(10)
            ],
            "feedback": {
                "total": len(feedback),
                "helpful": sum(1 for r in feedback if r["helpful"]),
            },
        }


class PostgresRepository:
    def __init__(self, database_url: str) -> None:
        from app.db.session import session_factory

        self._sessions = session_factory(database_url)

    def log_query(self, entry: QueryLog) -> None:
        from app.db import models

        record = {k: v for k, v in asdict(entry).items() if k != "created_at"}
        with self._sessions.begin() as session:
            session.add(models.QueryLog(**record))

    def save_feedback(self, entry: Feedback) -> int:
        from app.db import models

        record = {k: v for k, v in asdict(entry).items() if k != "created_at"}
        with self._sessions.begin() as session:
            row = models.Feedback(**record)
            session.add(row)
            session.flush()
            return row.id

    def usage_stats(self) -> dict:
        from sqlalchemy import func, select

        from app.db.models import Feedback as FeedbackRow
        from app.db.models import QueryLog as QueryLogRow

        with self._sessions() as session:
            total, avg_latency = session.execute(
                select(func.count(QueryLogRow.id), func.avg(QueryLogRow.latency_ms))
            ).one()
            by_status = dict(
                session.execute(
                    select(QueryLogRow.status, func.count()).group_by(QueryLogRow.status)
                ).all()
            )
            top = session.execute(
                select(QueryLogRow.normalized_query, func.count().label("n"))
                .group_by(QueryLogRow.normalized_query)
                .order_by(func.count().desc(), QueryLogRow.normalized_query)
                .limit(10)
            ).all()
            fb_total, fb_helpful = session.execute(
                select(
                    func.count(FeedbackRow.id),
                    func.count(FeedbackRow.id).filter(FeedbackRow.helpful.is_(True)),
                )
            ).one()
        return {
            "total_queries": total,
            "status_rates": {
                s: by_status.get(s, 0) / total for s in ("answer", "suggest", "not_found")
            }
            if total
            else {},
            "avg_latency_ms": float(avg_latency) if avg_latency is not None else None,
            "top_questions": [{"question": q, "count": n} for q, n in top],
            "feedback": {"total": fb_total, "helpful": fb_helpful},
        }


def build_repository(settings: "Settings") -> Repository:
    """Postgres when the data lives there, JSON lines for file-only runs."""
    if settings.data_source == "postgres":
        return PostgresRepository(settings.database_url)
    return JsonlRepository(settings.logs_dir)
