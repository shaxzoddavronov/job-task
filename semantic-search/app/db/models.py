"""SQLAlchemy models. PostgreSQL stores data, feedback and logs; search runs in RAM."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Answer(Base):
    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    category_id: Mapped[int | None] = mapped_column(Integer)
    topic_id: Mapped[int | None] = mapped_column(Integer)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime | None] = mapped_column(DateTime)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime)


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    answer_id: Mapped[int] = mapped_column(ForeignKey("answers.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    answer_id: Mapped[int | None] = mapped_column(ForeignKey("answers.id"))
    helpful: Mapped[bool] = mapped_column(Boolean)
    source: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    normalized_query: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16))
    answer_id: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
