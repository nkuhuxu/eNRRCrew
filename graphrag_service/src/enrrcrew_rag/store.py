from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pyarrow.parquet as parquet

from .schemas import BatchSummary, JobSummary, VersionSummary

ACTIVE_JOB_STATUSES = {
    "approved",
    "queued",
    "preparing",
    "indexing",
    "validating",
    "publishing",
}
RETRYABLE_BATCH_STATUSES = {"needs_review", "validated", "failed", "interrupted"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def release_is_valid(release_path: Path, expected_documents: int) -> bool:
    corpus_path = release_path / "input" / "documents.jsonl"
    documents_path = release_path / "output" / "documents.parquet"
    if not corpus_path.is_file() or not documents_path.is_file():
        return False
    corpus_count = sum(1 for line in corpus_path.read_text(encoding="utf-8").splitlines() if line)
    indexed_count = parquet.ParquetFile(documents_path).metadata.num_rows
    return corpus_count == expected_documents == indexed_count


class ControlStore:
    def __init__(self, database_path: Path, knowledge_root: Path) -> None:
        self.database_path = database_path
        self.knowledge_root = knowledge_root
        self._write_lock = threading.Lock()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                    source_file TEXT NOT NULL, status TEXT NOT NULL,
                    total INTEGER NOT NULL, accepted INTEGER NOT NULL,
                    duplicates INTEGER NOT NULL, conflicts INTEGER NOT NULL,
                    invalid INTEGER NOT NULL, issues_json TEXT NOT NULL,
                    documents_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, batch_id TEXT, status TEXT NOT NULL,
                    progress INTEGER NOT NULL, message TEXT NOT NULL DEFAULT '',
                    version TEXT, error TEXT, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, heartbeat_at TEXT
                );
                CREATE TABLE IF NOT EXISTS versions (
                    version TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    document_count INTEGER NOT NULL, release_path TEXT NOT NULL
                );
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "message" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN message TEXT NOT NULL DEFAULT ''"
                )
            if "heartbeat_at" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN heartbeat_at TEXT")
            connection.execute(
                "UPDATE jobs SET heartbeat_at=updated_at WHERE heartbeat_at IS NULL"
            )

    def create_batch(
        self,
        session_id: str,
        source_file: str,
        documents: list[dict[str, object]],
        issues: list[dict[str, object]],
        counts: dict[str, int],
    ) -> BatchSummary:
        batch_id = uuid4().hex
        total = len(documents)
        accepted = counts["accepted"]
        conflicts = counts["conflicts"]
        invalid = counts.get("invalid", 0)
        status = "needs_review" if accepted or conflicts else "validated"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO batches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    batch_id,
                    session_id,
                    Path(source_file).name,
                    status,
                    total,
                    accepted,
                    counts["duplicates"],
                    conflicts,
                    invalid,
                    json.dumps(issues, ensure_ascii=False),
                    json.dumps(documents, ensure_ascii=False),
                    utc_now(),
                ),
            )
        return self.get_batch(batch_id)

    def get_batch(self, batch_id: str) -> BatchSummary:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        if row is None:
            raise KeyError(batch_id)
        return BatchSummary(
            batch_id=row["batch_id"],
            status=row["status"],
            total=row["total"],
            accepted=row["accepted"],
            duplicates=row["duplicates"],
            conflicts=row["conflicts"],
            invalid=row["invalid"],
            issues=json.loads(row["issues_json"]),
        )

    def batch_documents(self, batch_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT documents_json FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        if row is None:
            raise KeyError(batch_id)
        return list(json.loads(row["documents_json"]))

    def update_batch_status(self, batch_id: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE batches SET status = ? WHERE batch_id = ?", (status, batch_id)
            )

    def create_job(self, batch_id: str | None = None) -> JobSummary:
        job_id = uuid4().hex
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO jobs
                (job_id, batch_id, status, progress, message, version, error,
                 created_at, updated_at, heartbeat_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    batch_id,
                    "queued",
                    1,
                    "Waiting for the indexing worker",
                    None,
                    None,
                    now,
                    now,
                    now,
                ),
            )
        return self.get_job(job_id)

    def create_queued_job(self, batch_id: str) -> JobSummary:
        job_id = uuid4().hex
        now = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            batch = connection.execute(
                "SELECT status FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise KeyError(batch_id)
            if batch["status"] not in RETRYABLE_BATCH_STATUSES:
                raise ValueError("Batch cannot be approved in its current state")
            active = connection.execute(
                f"SELECT 1 FROM jobs WHERE status IN ({','.join('?' for _ in ACTIVE_JOB_STATUSES)}) LIMIT 1",
                tuple(ACTIVE_JOB_STATUSES),
            ).fetchone()
            if active is not None:
                raise RuntimeError("Another indexing job is already running")
            connection.execute(
                "UPDATE batches SET status = ? WHERE batch_id = ?",
                ("queued", batch_id),
            )
            connection.execute(
                """INSERT INTO jobs
                (job_id, batch_id, status, progress, message, version, error,
                 created_at, updated_at, heartbeat_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    batch_id,
                    "queued",
                    1,
                    "Waiting for the indexing worker",
                    None,
                    None,
                    now,
                    now,
                    now,
                ),
            )
        return self.get_job(job_id)

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        progress: int,
        message: str = "",
        version: str | None = None,
        error: str | None = None,
    ) -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """UPDATE jobs
                SET status=?, progress=?, message=?, version=?, error=?,
                    updated_at=?, heartbeat_at=?
                WHERE job_id=?""",
                (status, progress, message, version, error, now, now, job_id),
            )

    def get_job(self, job_id: str) -> JobSummary:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        values = dict(row)
        created = datetime.fromisoformat(values["created_at"])
        values["elapsed_seconds"] = max(
            0, int((datetime.now(UTC) - created).total_seconds())
        )
        return JobSummary(**values)

    def interrupt_incomplete_jobs(self) -> int:
        now = utc_now()
        placeholders = ",".join("?" for _ in ACTIVE_JOB_STATUSES)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT job_id, batch_id FROM jobs WHERE status IN ({placeholders})",
                tuple(ACTIVE_JOB_STATUSES),
            ).fetchall()
            if not rows:
                return 0
            connection.execute(
                f"""UPDATE jobs
                SET status='interrupted',
                    progress=CASE WHEN progress >= 100 THEN 99 ELSE progress END,
                    message='Service restarted before indexing completed',
                    error='Indexing was interrupted by a service restart',
                    updated_at=?, heartbeat_at=?
                WHERE status IN ({placeholders})""",
                (now, now, *tuple(ACTIVE_JOB_STATUSES)),
            )
            batch_ids = [row["batch_id"] for row in rows if row["batch_id"]]
            if batch_ids:
                batch_placeholders = ",".join("?" for _ in batch_ids)
                connection.execute(
                    f"UPDATE batches SET status='interrupted' WHERE batch_id IN ({batch_placeholders})",
                    tuple(batch_ids),
                )
        return len(rows)

    def add_version(self, version: str, document_count: int, release_path: Path) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO versions VALUES (?, ?, ?, ?)",
                (version, utc_now(), document_count, str(release_path)),
            )

    def versions(self, active_version: str | None) -> list[VersionSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM versions ORDER BY created_at DESC"
            ).fetchall()
        return [
            VersionSummary(
                version=row["version"],
                created_at=row["created_at"],
                document_count=row["document_count"],
                active=row["version"] == active_version,
                valid=release_is_valid(Path(row["release_path"]), row["document_count"]),
            )
            for row in rows
        ]

    @contextmanager
    def writer(self) -> Iterator[None]:
        if not self._write_lock.acquire(blocking=False):
            raise RuntimeError("Another indexing job already holds the writer lock")
        try:
            yield
        finally:
            self._write_lock.release()


def read_active(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")))


def write_active(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)
