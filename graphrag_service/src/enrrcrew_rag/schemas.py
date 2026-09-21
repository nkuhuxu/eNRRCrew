from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CorpusDocument(BaseModel):
    document_id: str
    title: str = Field(min_length=3, max_length=500)
    abstract: str = Field(min_length=100, max_length=20_000)
    publication_year: int | None = None
    doi: str | None = None
    content_sha256: str
    source_file: str
    source_sheet: str | None = None
    source_row: int | None = None
    revision: int = Field(default=1, ge=1)

    @field_validator("publication_year")
    @classmethod
    def valid_year(cls, value: int | None) -> int | None:
        if value is not None and not 1800 <= value <= datetime.now(UTC).year + 1:
            raise ValueError("publication year is outside the accepted range")
        return value


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20_000)
    mode: Literal["local", "global"] = "local"
    community_level: int = Field(default=0, ge=0, le=10)
    response_type: str = Field(default="multiple paragraphs", min_length=1, max_length=100)


class RagCitation(BaseModel):
    document_id: str
    title: str
    doi: str | None = None
    publication_year: int | None = None
    excerpt: str | None = None


class RagQueryResult(BaseModel):
    answer: str
    mode: Literal["local", "global"]
    index_version: str
    citations: list[RagCitation] = Field(default_factory=list)
    elapsed_ms: int


class BatchDocument(BaseModel):
    title: str
    abstract: str
    doi: str | None = None
    publication_year: int | None = None
    source_row: int | None = None


class BatchCreateRequest(BaseModel):
    session_id: str = Field(pattern=r"^[A-Za-z0-9-]+$")
    source_file: str = Field(min_length=1, max_length=260)
    documents: list[BatchDocument] = Field(min_length=1, max_length=1000)


class BatchSummary(BaseModel):
    batch_id: str
    status: str
    total: int
    accepted: int
    duplicates: int
    conflicts: int
    invalid: int
    issues: list[dict[str, object]] = Field(default_factory=list)


class JobSummary(BaseModel):
    job_id: str
    batch_id: str | None = None
    status: str
    progress: int = Field(ge=0, le=100)
    message: str = ""
    version: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str
    heartbeat_at: str | None = None
    elapsed_seconds: int = Field(default=0, ge=0)


class VersionSummary(BaseModel):
    version: str
    created_at: str
    document_count: int
    active: bool
    valid: bool = True
