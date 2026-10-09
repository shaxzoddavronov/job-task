"""Pydantic response models (plan section 12).

Every user-visible string is in the document's original script. Missing values
are null, never invented.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Field_(BaseModel):
    """A detail with a note of how it was found."""

    value: str | None = None
    method: str | None = None  # "regex" | "llm" | None


class DateField(Field_):
    raw: str | None = None  # the matched text, in the original script


class Signatory(BaseModel):
    name: str | None = None
    position: str | None = None
    method: str | None = None


class Details(BaseModel):
    organization: Field_ = Field_()
    document_number: Field_ = Field_()
    document_date: DateField = DateField()
    signatory: Signatory = Signatory()


class Task(BaseModel):
    id: int
    task: str
    deadline_raw: str | None = None
    deadline_date: str | None = None
    deadline_type: str = "unknown"
    executors: list[str] = Field(default_factory=list)
    page: int = 1
    source: str = "clause"  # clause | table_row | paragraph
    grounded: bool = True


class SourceInfo(BaseModel):
    pages: int
    scanned_pages: int
    script: str


class Timings(BaseModel):
    extract: int = 0
    llm: int = 0
    total: int = 0


class Analysis(BaseModel):
    model: str
    source: SourceInfo
    details: Details
    tasks: list[Task] = Field(default_factory=list)
    timings_ms: Timings = Timings()
