from __future__ import annotations

import time
from pathlib import Path
from threading import Event

import pandas as pd
from fastapi.testclient import TestClient

from enrrcrew_rag.app import create_app
from enrrcrew_rag.config import ServiceSettings
from enrrcrew_rag.corpus import make_document
from enrrcrew_rag.schemas import BatchDocument, RagQueryResult
from enrrcrew_rag.store import read_active, write_active


def settings(tmp_path: Path) -> ServiceSettings:
    return ServiceSettings(
        project_root=tmp_path,
        knowledge_root=tmp_path / "knowledge",
        service_token="service-secret",
        admin_token="admin-secret",
    )


def headers(admin: bool = False) -> dict[str, str]:
    value = {
        "X-ENRRCREW-SERVICE-TOKEN": "service-secret",
        "X-ENRRCREW-UPSTREAM-KEY": "upstream-secret",
        "X-ENRRCREW-UPSTREAM-BASE-URL": "https://api.test/v1",
    }
    if admin:
        value["X-ENRRCREW-ADMIN-TOKEN"] = "admin-secret"
    return value


def payload(doi: str = "10.test/new", suffix: str = "new") -> dict[str, object]:
    return {
        "session_id": "session-one",
        "source_file": "../papers.csv",
        "documents": [
            {
                "title": f"New catalyst {suffix}",
                "abstract": (f"Scientific abstract {suffix} about nitrogen reduction. " * 5),
                "doi": doi,
                "publication_year": 2025,
                "source_row": 2,
            }
        ],
    }


def test_auth_health_query_and_redacted_failure(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path))
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/v1/query", json={"question": "test"}).status_code == 401

    app.state.runtime.runner.query = lambda *args: RagQueryResult(
        answer="answer",
        mode="local",
        index_version="v1",
        elapsed_ms=1,
    )
    response = client.post("/v1/query", headers=headers(), json={"question": "test"})
    assert response.status_code == 200
    assert response.json()["answer"] == "answer"

    def failed(*args: object):
        raise RuntimeError("upstream-secret D:\\private")

    app.state.runtime.runner.query = failed
    response = client.post("/v1/query", headers=headers(), json={"question": "test"})
    assert response.status_code == 503
    assert response.json() == {"detail": "GraphRAG query failed"}
    assert "secret" not in response.text


def test_batch_validation_approval_publish_and_duplicate(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    app = create_app(configured)
    client = TestClient(app)
    created = client.post("/v1/batches", headers=headers(), json=payload())
    assert created.status_code == 200
    batch = created.json()
    assert batch["accepted"] == 1
    assert batch["status"] == "needs_review"
    batch_id = batch["batch_id"]
    assert client.get(f"/v1/batches/{batch_id}", headers=headers()).status_code == 200
    assert client.get("/v1/batches/missing", headers=headers()).status_code == 404

    rejected = client.post(f"/v1/batches/{batch_id}/approve", headers=headers())
    assert rejected.status_code == 403

    def publish(documents, *args, **kwargs):
        release = configured.knowledge_root / "releases" / "v-test"
        release.mkdir(parents=True, exist_ok=True)
        app.state.runtime.store.add_version("v-test", len(documents), release)
        write_active(configured.active_path, {"version": "v-test", "previous_version": None})
        return "v-test"

    app.state.runtime.runner.build_release = publish
    approved = client.post(
        f"/v1/batches/{batch_id}/approve", headers=headers(admin=True)
    )
    assert approved.status_code == 200
    job_id = approved.json()["job_id"]
    for _ in range(100):
        job = client.get(f"/v1/jobs/{job_id}", headers=headers()).json()
        if job["status"] not in {
            "approved",
            "queued",
            "preparing",
            "indexing",
            "validating",
            "publishing",
        }:
            break
        time.sleep(0.01)
    assert job["status"] == "published"
    assert job["progress"] == 100
    assert job["message"] == "Index release published and activated"
    assert job["elapsed_seconds"] >= 0
    assert (configured.knowledge_root / "corpus" / "documents.jsonl").is_file()

    duplicate = client.post("/v1/batches", headers=headers(), json=payload()).json()
    assert duplicate["duplicates"] == 1
    assert duplicate["accepted"] == 0
    conflict = client.post(
        "/v1/batches", headers=headers(), json=payload(suffix="changed")
    ).json()
    assert conflict["conflicts"] == 1
    no_docs = client.post(
        f"/v1/batches/{duplicate['batch_id']}/approve", headers=headers(admin=True)
    )
    assert no_docs.status_code == 409
    assert client.get("/v1/jobs/missing", headers=headers()).status_code == 404


def test_versions_activate_and_rollback(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    app = create_app(configured)
    client = TestClient(app)
    for version in ("v1", "v2"):
        release = configured.knowledge_root / "releases" / version
        (release / "input").mkdir(parents=True, exist_ok=True)
        (release / "output").mkdir(parents=True, exist_ok=True)
        documents = [
            make_document(
                BatchDocument(
                    title=f"Test document {index}",
                    abstract=(f"Scientific validation abstract number {index}. " * 5),
                    doi=f"10.test/{version}/{index}",
                    publication_year=2025,
                ),
                "test.csv",
            )
            for index in range(10)
        ]
        (release / "input" / "documents.jsonl").write_text(
            "".join(item.model_dump_json() + "\n" for item in documents),
            encoding="utf-8",
        )
        pd.DataFrame({"id": range(10)}).to_parquet(
            release / "output" / "documents.parquet", index=False
        )
        app.state.runtime.store.add_version(version, 10, release)
    write_active(configured.active_path, {"version": "v1", "previous_version": None})
    versions = client.get("/v1/versions", headers=headers()).json()
    assert next(item for item in versions if item["version"] == "v1")["active"] is True
    assert client.post("/v1/versions/v2/activate", headers=headers()).status_code == 403
    activated = client.post(
        "/v1/versions/v2/activate", headers=headers(admin=True)
    )
    assert activated.json()["version"] == "v2"
    assert read_active(configured.active_path)["previous_version"] == "v1"
    rolled_back = client.post("/v1/versions/rollback", headers=headers(admin=True))
    assert rolled_back.json()["version"] == "v1"
    missing = client.post("/v1/versions/none/activate", headers=headers(admin=True))
    assert missing.status_code == 404


def test_concurrent_or_repeated_approval_is_rejected(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    app = create_app(configured)
    client = TestClient(app)
    first = client.post("/v1/batches", headers=headers(), json=payload()).json()
    second = client.post(
        "/v1/batches",
        headers=headers(),
        json=payload(doi="10.test/second", suffix="second"),
    ).json()
    started = Event()
    release = Event()

    def slow_publish(documents, *args, **kwargs):
        started.set()
        release.wait(timeout=5)
        return "v-slow"

    app.state.runtime.runner.build_release = slow_publish
    approved = client.post(
        f"/v1/batches/{first['batch_id']}/approve", headers=headers(admin=True)
    )
    assert approved.status_code == 200
    assert started.wait(timeout=2)
    repeated = client.post(
        f"/v1/batches/{first['batch_id']}/approve", headers=headers(admin=True)
    )
    concurrent = client.post(
        f"/v1/batches/{second['batch_id']}/approve", headers=headers(admin=True)
    )
    assert repeated.status_code == 409
    assert concurrent.status_code == 409
    assert concurrent.json()["detail"] == "Another indexing job is already running"
    release.set()
