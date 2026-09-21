from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from enrrcrew.config import AppSettings, SessionWorkspace
from enrrcrew.services.knowledge import (
    KnowledgeService,
    detect_columns,
    frame_to_documents,
    read_upload,
    safe_csv,
)

ACTIVE_JOB_STATUSES = {
    "approved",
    "queued",
    "preparing",
    "indexing",
    "validating",
    "publishing",
}
RETRYABLE_BATCH_STATUSES = {"needs_review", "validated", "failed", "interrupted"}


def _client(settings: AppSettings, admin_token: str = "") -> KnowledgeService:
    return KnowledgeService(
        settings.rag_service_url,
        settings.rag_service_token,
        settings.api_key,
        settings.base_url,
        admin_token,
    )


def _mapping_selectors(frame: pd.DataFrame) -> dict[str, str | None]:
    detected = detect_columns([str(item) for item in frame.columns])
    choices = ["— Not provided —", *[str(item) for item in frame.columns]]
    labels = {
        "title": "Title column (required)",
        "abstract": "Abstract column (required)",
        "doi": "DOI column",
        "publication_year": "Publication year column",
    }
    mapping: dict[str, str | None] = {}
    for field, label in labels.items():
        default = detected[field]
        index = choices.index(default) if default in choices else 0
        selected = st.selectbox(label, choices, index=index, key=f"kb_column_{field}")
        mapping[field] = None if selected == choices[0] else selected
    return mapping


def _render_batch(batch: dict[str, object]) -> None:
    columns = st.columns(5)
    for column, label, key in zip(
        columns,
        ("New", "Duplicate", "Conflict", "Invalid", "Total"),
        ("accepted", "duplicates", "conflicts", "invalid", "total"),
        strict=True,
    ):
        column.metric(label, batch.get(key, 0))
    st.caption(f"Batch `{batch['batch_id']}` · status: `{batch['status']}`")
    issues = list(batch.get("issues", []))
    if issues:
        st.dataframe(issues, use_container_width=True)
        st.download_button(
            "Download validation report",
            safe_csv(issues),
            file_name=f"batch-{batch['batch_id']}-validation.csv",
            mime="text/csv",
        )


def _knowledge_next_action(
    batch: dict[str, object] | None,
    job: dict[str, object] | None,
) -> tuple[str, str]:
    if _job_is_active(job):
        return (
            "Wait for indexing to finish",
            "The page refreshes progress automatically. Keep the launch terminal open; "
            "do not revalidate, approve again, or switch versions.",
        )
    if job and str(job.get("status")) == "published":
        return (
            "Verify the new version",
            "Select Load index versions, confirm that the latest release is active and "
            "valid, then retrieve the new paper in Dialogue & retrieval.",
        )
    if job and str(job.get("status")) in {"failed", "interrupted"}:
        return (
            "Review the failure before retrying",
            "Confirm that the API, network, and launch terminal are available. After "
            "correcting the issue, approve the current batch once.",
        )
    if batch and int(batch.get("accepted", 0)) == 0:
        return (
            "No new index is required",
            "This batch contains no publishable new documents. Duplicates are skipped; "
            "correct conflicts or invalid rows in the source file.",
        )
    if batch and str(batch.get("status")) in RETRYABLE_BATCH_STATUSES:
        return (
            "Review and approve the batch",
            "Check New, Duplicate, Conflict, and Invalid. If the batch is ready, enter "
            "the administrator token and approve it once.",
        )
    return (
        "Upload and validate a literature table",
        "Choose CSV, XLS, or XLSX, confirm the title and abstract column mapping, then "
        "select Validate and submit for review.",
    )


def _render_knowledge_tutorial(
    batch: dict[str, object] | None,
    job: dict[str, object] | None,
) -> None:
    title, guidance = _knowledge_next_action(batch, job)
    with st.expander("How to use · Knowledge update runbook", expanded=True):
        st.info(f"Recommended next step: **{title}**. {guidance}")
        st.markdown(
            """
**Standard workflow**

`Upload file` → `Confirm column mapping` → `Validate` → `Review validation counts`
→ `Approve once` → `Wait for published` → `Load index versions` → `Verify retrieval`

| Control | Use it when | Do not use it when |
|---|---|---|
| **Validate and submit for review** | The file and column mapping are ready | An indexing job is running |
| **Approve and start indexing** | `New > 0` and the review is complete; click once | The job is queued/indexing, or every row is a duplicate |
| **Refresh job status** | You want an immediate refresh; the page also refreshes automatically | Repeated rapid clicks are unnecessary |
| **Load index versions** | Publishing has finished, or you are preparing to switch versions | This control does not start indexing |
| **Activate selected version** | You intentionally want to switch to another valid version | The selected version is already active |
| **Rollback to previous version** | You need to undo the latest version switch | You only want to inspect version history |

**Completion check:** The job shows `published`, and the target release shows both
`active = true` and `valid = true`.
"""
        )


def _job_is_active(job: dict[str, object] | None) -> bool:
    return bool(job and str(job.get("status", "")) in ACTIVE_JOB_STATUSES)


def _format_elapsed(seconds: object) -> str:
    total = max(0, int(seconds or 0))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes:d}m {seconds:02d}s"


def _seconds_since(value: object) -> int | None:
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(str(value))
        return max(0, int((datetime.now(UTC) - timestamp).total_seconds()))
    except ValueError:
        return None


def _refresh_job(settings: AppSettings, job: dict[str, object]) -> dict[str, object]:
    updated = _client(settings).get_job(str(job["job_id"]))
    st.session_state.kb_job = updated
    if not _job_is_active(updated) and st.session_state.get("kb_batch"):
        batch_id = str(st.session_state.kb_batch["batch_id"])
        st.session_state.kb_batch = _client(settings).get_batch(batch_id)
    return updated


def _render_job_panel(settings: AppSettings) -> None:
    job = st.session_state.get("kb_job")
    if not job:
        return
    if _job_is_active(job):
        try:
            job = _refresh_job(settings, job)
        except Exception as exc:
            st.warning(str(exc))

    progress = int(job.get("progress", 0))
    status = str(job.get("status", "unknown"))
    message = str(job.get("message", ""))
    st.progress(progress, text=f"{status}: {message}" if message else status)
    stage, elapsed, heartbeat = st.columns(3)
    stage.metric("Stage", status)
    elapsed.metric("Elapsed", _format_elapsed(job.get("elapsed_seconds", 0)))
    heartbeat_age = _seconds_since(job.get("heartbeat_at"))
    heartbeat.metric(
        "Last activity",
        "unknown" if heartbeat_age is None else f"{heartbeat_age}s ago",
    )
    st.caption(f"Job `{job['job_id']}`")
    if status in {"failed", "interrupted"}:
        st.error(str(job.get("error") or message or "Indexing did not complete"))
    elif status == "published":
        st.success(f"Published index version `{job.get('version')}`.")

    if st.button("Refresh job status", key="kb_refresh_job"):
        try:
            _refresh_job(settings, job)
            st.rerun(scope="fragment")
        except Exception as exc:
            st.error(str(exc))


@st.fragment(run_every=3)
def _render_job_monitor(settings: AppSettings) -> None:
    _render_job_panel(settings)


def render_knowledge_update(settings: AppSettings, workspace: SessionWorkspace) -> None:
    st.subheader("Knowledge base update")
    st.info(
        "Uploads are validated and queued for administrator review. Publishing creates a new "
        "immutable GraphRAG release; the current release stays queryable during indexing."
    )
    current_batch = st.session_state.get("kb_batch")
    current_job = st.session_state.get("kb_job")
    _render_knowledge_tutorial(current_batch, current_job)
    job_active = _job_is_active(current_job)
    upload = st.file_uploader(
        "Literature table",
        type=["csv", "xls", "xlsx"],
        help="Maximum 20 MB and 1,000 rows. Title and abstract are required.",
        key="kb_upload",
    )
    if upload is not None:
        try:
            payload = upload.getvalue()
            upload_dir = workspace.root / "knowledge_uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            safe_name = Path(upload.name).name
            upload_path = upload_dir / safe_name
            upload_path.write_bytes(payload)
            frame = read_upload(safe_name, payload)
            st.caption(f"Loaded {len(frame)} rows from `{safe_name}`")
            st.dataframe(frame.head(20), use_container_width=True)
            mapping = _mapping_selectors(frame)
            if st.button(
                "Validate and submit for review",
                key="kb_submit",
                disabled=job_active,
            ):
                documents = frame_to_documents(frame, mapping)
                with st.spinner("Validating the batch…"):
                    batch = _client(settings).create_batch(
                        workspace.session_id, safe_name, documents
                    )
                st.session_state.kb_batch = batch
                st.success("Batch submitted for administrator review.")
        except Exception as exc:
            st.error(str(exc))

    batch = st.session_state.get("kb_batch")
    if batch:
        _render_batch(batch)

    st.markdown("#### Administrator actions")
    admin_token = st.text_input(
        "Administrator token",
        type="password",
        value="",
        help="Used for this request only and never written to disk.",
        key="kb_admin_token",
    )
    if batch and st.button(
        "Approve and start indexing",
        disabled=(
            not admin_token
            or int(batch.get("accepted", 0)) == 0
            or job_active
            or str(batch.get("status", "")) not in RETRYABLE_BATCH_STATUSES
        ),
        key="kb_approve",
    ):
        try:
            job = _client(settings).approve(str(batch["batch_id"]), admin_token)
            st.session_state.kb_job = job
            st.session_state.kb_batch = {**batch, "status": "queued"}
            st.success("Indexing job accepted.")
        except Exception as exc:
            st.error(str(exc))

    _render_job_monitor(settings)

    if st.button("Load index versions", key="kb_versions"):
        try:
            st.session_state.kb_versions_data = _client(settings).versions()
        except Exception as exc:
            st.error(str(exc))
    versions = st.session_state.get("kb_versions_data", [])
    if versions:
        st.dataframe(versions, use_container_width=True)
        valid_versions = [
            str(item["version"]) for item in versions if item.get("valid", True)
        ]
        invalid_count = len(versions) - len(valid_versions)
        if invalid_count:
            st.warning(
                f"{invalid_count} release(s) failed integrity validation and cannot be activated."
            )
        selected = (
            st.selectbox(
                "Version to activate",
                valid_versions,
                key="kb_selected_version",
            )
            if valid_versions
            else None
        )
        left, right = st.columns(2)
        if left.button(
            "Activate selected version",
            disabled=not admin_token or selected is None,
        ):
            try:
                assert selected is not None
                _client(settings).activate(selected, admin_token)
                st.success(f"Activated `{selected}`.")
            except Exception as exc:
                st.error(str(exc))
        if right.button("Rollback to previous version", disabled=not admin_token):
            try:
                result = _client(settings).rollback(admin_token)
                st.success(f"Rolled back to `{result['version']}`.")
            except Exception as exc:
                st.error(str(exc))
