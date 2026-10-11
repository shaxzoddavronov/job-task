"""Storage for query logs and feedback (never used for search).

`Repository` is the interface the API depends on. `JsonlRepository` writes JSON lines to
data/logs/ until PostgreSQL arrives (Phase 9b); the records mirror the planned tables so the
Postgres implementation can replace it without touching the API.
"""

import json
import threading
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


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
