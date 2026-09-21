from __future__ import annotations

import hmac
import logging
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, status

from .config import ServiceSettings
from .corpus import make_document, merge_documents, read_jsonl, write_jsonl
from .runner import GraphRunner
from .schemas import (
    BatchCreateRequest,
    BatchSummary,
    CorpusDocument,
    JobSummary,
    RagQueryRequest,
    RagQueryResult,
    VersionSummary,
)
from .store import ControlStore, read_active, release_is_valid, write_active

LOGGER = logging.getLogger("enrrcrew_rag")


def _configure_logging(knowledge_root: Path) -> None:
    log_path = knowledge_root / "state" / "service.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(log_path.resolve())
    if any(getattr(handler, "baseFilename", None) == resolved for handler in LOGGER.handlers):
        return
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


class ServiceRuntime:
    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self.store = ControlStore(settings.database_path, settings.knowledge_root)
        interrupted = self.store.interrupt_incomplete_jobs()
        if interrupted:
            LOGGER.warning(
                "Marked %s unfinished indexing job(s) as interrupted after service startup",
                interrupted,
            )
        self.runner = GraphRunner(settings.knowledge_root, self.store)


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    configured = settings or ServiceSettings.from_environment()
    _configure_logging(configured.knowledge_root)
    runtime = ServiceRuntime(configured)
    app = FastAPI(title="eNRRCrew GraphRAG service", version="0.5.1")
    app.state.runtime = runtime

    def active_corpus() -> list[CorpusDocument]:
        active = read_active(configured.active_path)
        if active:
            release_corpus = (
                configured.knowledge_root
                / "releases"
                / str(active["version"])
                / "input"
                / "documents.jsonl"
            )
            if release_corpus.is_file():
                return read_jsonl(release_corpus)
        return read_jsonl(configured.corpus_path)

    def service_auth(
        x_enrrcrew_service_token: str = Header(default=""),
    ) -> None:
        expected = configured.service_token
        if not expected or not hmac.compare_digest(x_enrrcrew_service_token, expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    def admin_auth(
        x_enrrcrew_admin_token: str = Header(default=""),
    ) -> None:
        expected = configured.admin_token
        if not expected or not hmac.compare_digest(x_enrrcrew_admin_token, expected):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator authorization failed")

    @app.get("/health")
    def health() -> dict[str, object]:
        active = read_active(configured.active_path)
        return {
            "status": "ok",
            "version": "0.5.1",
            "active_index": active.get("version") if active else None,
        }

    @app.post("/v1/query", response_model=RagQueryResult, dependencies=[Depends(service_auth)])
    def query(
        request: RagQueryRequest,
        x_enrrcrew_upstream_key: str = Header(default=""),
        x_enrrcrew_upstream_base_url: str = Header(default="https://api.openai.com/v1"),
    ) -> RagQueryResult:
        try:
            return runtime.runner.query(
                request, x_enrrcrew_upstream_key, x_enrrcrew_upstream_base_url
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except RuntimeError as exc:
            LOGGER.error("GraphRAG query failed (%s)", type(exc).__name__)
            raise HTTPException(status_code=503, detail="GraphRAG query failed") from None

    @app.post(
        "/v1/batches",
        response_model=BatchSummary,
        dependencies=[Depends(service_auth)],
    )
    def create_batch(request: BatchCreateRequest) -> BatchSummary:
        current = active_corpus()
        incoming: list[CorpusDocument] = []
        issues: list[dict[str, object]] = []
        invalid = 0
        for index, raw in enumerate(request.documents, start=1):
            try:
                incoming.append(
                    make_document(
                        raw,
                        Path(request.source_file).name,
                        source_row=raw.source_row or index + 1,
                    )
                )
            except ValueError as exc:
                invalid += 1
                issues.append(
                    {"source_row": raw.source_row or index + 1, "kind": "invalid", "message": str(exc)}
                )
        merged, merge_issues, counts = merge_documents(current, incoming)
        issues.extend(merge_issues)
        current_ids = {item.document_id for item in current}
        accepted = [
            item for item in merged if item.document_id not in current_ids
        ]
        counts["invalid"] = invalid
        return runtime.store.create_batch(
            request.session_id,
            request.source_file,
            [item.model_dump(mode="json") for item in accepted],
            issues,
            counts,
        )

    @app.get(
        "/v1/batches/{batch_id}",
        response_model=BatchSummary,
        dependencies=[Depends(service_auth)],
    )
    def get_batch(batch_id: str) -> BatchSummary:
        try:
            return runtime.store.get_batch(batch_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Batch not found") from None

    def index_batch(
        batch_id: str,
        job_id: str,
        api_key: str,
        base_url: str,
    ) -> None:
        def report_progress(stage: str, progress: int, message: str) -> None:
            runtime.store.update_job(
                job_id,
                status=stage,
                progress=progress,
                message=message,
            )
            runtime.store.update_batch_status(batch_id, stage)

        try:
            with runtime.store.writer():
                report_progress("preparing", 5, "Preparing the canonical corpus")
                current = active_corpus()
                incoming = [
                    CorpusDocument.model_validate(item)
                    for item in runtime.store.batch_documents(batch_id)
                ]
                merged, _, _ = merge_documents(current, incoming)
                version = runtime.runner.build_release(
                    merged,
                    api_key,
                    base_url,
                    update=bool(read_active(configured.active_path)),
                    progress_callback=report_progress,
                )
                try:
                    write_jsonl(configured.corpus_path, merged)
                except OSError as exc:
                    LOGGER.error("Canonical corpus mirror update failed (%s)", type(exc).__name__)
                runtime.store.update_job(
                    job_id,
                    status="published",
                    progress=100,
                    message="Index release published and activated",
                    version=version,
                )
                runtime.store.update_batch_status(batch_id, "published")
        except Exception:
            LOGGER.exception("Indexing job %s failed", job_id)
            runtime.store.update_job(
                job_id,
                status="failed",
                progress=100,
                message="Indexing failed",
                error="Indexing failed; inspect the local service log",
            )
            runtime.store.update_batch_status(batch_id, "failed")

    @app.post(
        "/v1/batches/{batch_id}/approve",
        response_model=JobSummary,
        dependencies=[Depends(service_auth), Depends(admin_auth)],
    )
    def approve_batch(
        batch_id: str,
        x_enrrcrew_upstream_key: str = Header(default=""),
        x_enrrcrew_upstream_base_url: str = Header(default="https://api.openai.com/v1"),
    ) -> JobSummary:
        try:
            batch = runtime.store.get_batch(batch_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Batch not found") from None
        if batch.accepted == 0:
            raise HTTPException(status_code=409, detail="Batch contains no publishable documents")
        try:
            job = runtime.store.create_queued_job(batch_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except RuntimeError:
            raise HTTPException(
                status_code=409,
                detail="Another indexing job is already running",
            ) from None
        threading.Thread(
            target=index_batch,
            args=(batch_id, job.job_id, x_enrrcrew_upstream_key, x_enrrcrew_upstream_base_url),
            daemon=True,
            name=f"rag-index-{job.job_id[:8]}",
        ).start()
        return job

    @app.get(
        "/v1/jobs/{job_id}",
        response_model=JobSummary,
        dependencies=[Depends(service_auth)],
    )
    def get_job(job_id: str) -> JobSummary:
        try:
            return runtime.store.get_job(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found") from None

    @app.get(
        "/v1/versions",
        response_model=list[VersionSummary],
        dependencies=[Depends(service_auth)],
    )
    def versions() -> list[VersionSummary]:
        active = read_active(configured.active_path)
        return runtime.store.versions(str(active["version"]) if active else None)

    def activate_version(version: str) -> dict[str, object]:
        release = configured.knowledge_root / "releases" / version
        if not release.is_dir():
            raise HTTPException(status_code=404, detail="Version not found")
        version_record = next(
            (item for item in runtime.store.versions(None) if item.version == version), None
        )
        if version_record is None or not release_is_valid(
            release, version_record.document_count
        ):
            raise HTTPException(status_code=409, detail="Version failed release validation")
        active = read_active(configured.active_path)
        write_active(
            configured.active_path,
            {
                "version": version,
                "output_dir": "output",
                "previous_version": active.get("version") if active else None,
            },
        )
        release_corpus = release / "input" / "documents.jsonl"
        if release_corpus.is_file():
            try:
                write_jsonl(configured.corpus_path, read_jsonl(release_corpus))
            except OSError as exc:
                LOGGER.error("Canonical corpus mirror update failed (%s)", type(exc).__name__)
        return {"status": "activated", "version": version}

    @app.post(
        "/v1/versions/{version}/activate",
        dependencies=[Depends(service_auth), Depends(admin_auth)],
    )
    def activate(version: str) -> dict[str, object]:
        return activate_version(version)

    @app.post(
        "/v1/versions/rollback",
        dependencies=[Depends(service_auth), Depends(admin_auth)],
    )
    def rollback() -> dict[str, object]:
        active = read_active(configured.active_path)
        previous = active.get("previous_version") if active else None
        if not previous:
            raise HTTPException(status_code=409, detail="No previous version is available")
        return activate_version(str(previous))

    return app


app = create_app()
