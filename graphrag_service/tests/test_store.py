from __future__ import annotations

from pathlib import Path

import pytest

from enrrcrew_rag.store import ControlStore, read_active, write_active


def test_store_batch_job_version_and_atomic_pointer(tmp_path: Path) -> None:
    store = ControlStore(tmp_path / "state" / "control.sqlite", tmp_path)
    batch = store.create_batch(
        "session-one",
        "../papers.csv",
        [{"document_id": "doi:1"}],
        [{"kind": "duplicate"}],
        {"accepted": 1, "duplicates": 1, "conflicts": 0, "invalid": 0},
    )
    assert batch.status == "needs_review"
    assert batch.total == 1
    assert store.batch_documents(batch.batch_id)[0]["document_id"] == "doi:1"
    store.update_batch_status(batch.batch_id, "approved")
    assert store.get_batch(batch.batch_id).status == "approved"
    with pytest.raises(KeyError):
        store.get_batch("missing")
    with pytest.raises(KeyError):
        store.batch_documents("missing")

    job = store.create_job(batch.batch_id)
    assert job.status == "queued"
    assert job.message == "Waiting for the indexing worker"
    assert job.heartbeat_at
    store.update_job(job.job_id, status="published", progress=100, version="v1")
    assert store.get_job(job.job_id).version == "v1"
    with pytest.raises(KeyError):
        store.get_job("missing")

    release = tmp_path / "releases" / "v1"
    release.mkdir(parents=True)
    store.add_version("v1", 3, release)
    versions = store.versions("v1")
    assert versions[0].active is True
    assert versions[0].document_count == 3

    active = tmp_path / "active.json"
    write_active(active, {"version": "v1"})
    assert read_active(active) == {"version": "v1"}
    assert read_active(tmp_path / "none.json") is None


def test_single_writer_rejects_concurrent_publication(tmp_path: Path) -> None:
    store = ControlStore(tmp_path / "control.sqlite", tmp_path)
    with store.writer(), pytest.raises(RuntimeError, match="writer lock"), store.writer():
        pass


def test_queued_job_is_atomic_and_restart_marks_it_interrupted(tmp_path: Path) -> None:
    store = ControlStore(tmp_path / "control.sqlite", tmp_path)
    first = store.create_batch(
        "session-one",
        "papers.csv",
        [{"document_id": "doi:1"}],
        [],
        {"accepted": 1, "duplicates": 0, "conflicts": 0, "invalid": 0},
    )
    second = store.create_batch(
        "session-two",
        "papers.csv",
        [{"document_id": "doi:2"}],
        [],
        {"accepted": 1, "duplicates": 0, "conflicts": 0, "invalid": 0},
    )
    job = store.create_queued_job(first.batch_id)
    assert store.get_batch(first.batch_id).status == "queued"
    with pytest.raises(RuntimeError, match="already running"):
        store.create_queued_job(second.batch_id)
    store.update_job(
        job.job_id,
        status="indexing",
        progress=50,
        message="Extracting entities",
    )
    assert store.interrupt_incomplete_jobs() == 1
    interrupted = store.get_job(job.job_id)
    assert interrupted.status == "interrupted"
    assert interrupted.progress == 50
    assert "restart" in interrupted.error
    assert store.get_batch(first.batch_id).status == "interrupted"
    retry = store.create_queued_job(first.batch_id)
    assert retry.status == "queued"
