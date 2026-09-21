from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

import enrrcrew.services.knowledge as knowledge
import enrrcrew.ui.knowledge as knowledge_ui
from enrrcrew.services.knowledge import (
    KnowledgeService,
    detect_columns,
    frame_to_documents,
    read_upload,
    safe_csv,
    safe_csv_cell,
)


class Response:
    def __init__(
        self,
        payload: object,
        status_code: int = 200,
        *,
        invalid_json: bool = False,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.invalid_json = invalid_json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://service/test")
            response = (
                httpx.Response(self.status_code, request=request, content=b"bad")
                if self.invalid_json
                else httpx.Response(self.status_code, request=request, json=self.payload)
            )
            raise httpx.HTTPStatusError("bad", request=request, response=response)

    def json(self):
        if self.invalid_json:
            raise ValueError("invalid")
        return self.payload


def test_upload_reading_column_detection_and_document_mapping(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "Article Title": ["Paper A", "Paper B"],
            "摘要": ["A" * 120, "B" * 120],
            "DOI": ["10.test/a", None],
            "年份": [2024, "not-a-year"],
        }
    )
    csv_payload = frame.to_csv(index=False).encode("utf-8")
    loaded = read_upload("papers.csv", csv_payload)
    mapping = detect_columns([str(item) for item in loaded.columns])
    assert mapping == {
        "title": "Article Title",
        "abstract": "摘要",
        "doi": "DOI",
        "publication_year": "年份",
    }
    documents = frame_to_documents(loaded, mapping)
    assert documents[0]["publication_year"] == 2024
    assert documents[1]["publication_year"] == -1
    assert documents[1]["doi"] is None
    assert documents[0]["source_row"] == 2

    stream = io.BytesIO()
    frame.iloc[:1].to_excel(stream, index=False, engine="openpyxl")
    assert len(read_upload("papers.xlsx", stream.getvalue())) == 1
    with pytest.raises(ValueError, match="Only CSV"):
        read_upload("papers.xlsm", b"payload")
    monkeypatch.setattr(knowledge, "MAX_UPLOAD_BYTES", 2)
    with pytest.raises(ValueError, match="20 MB"):
        read_upload("papers.csv", b"too large")


def test_upload_limits_required_mapping_and_formula_safe_csv(monkeypatch) -> None:
    frame = pd.DataFrame({"Title": ["A"], "Abstract": ["B"]})
    with pytest.raises(ValueError, match="required"):
        frame_to_documents(frame, {"title": "Title", "abstract": None})
    monkeypatch.setattr(knowledge, "MAX_UPLOAD_ROWS", 0)
    with pytest.raises(ValueError, match="1,000"):
        read_upload("papers.csv", frame.to_csv(index=False).encode())
    assert safe_csv_cell("=2+2") == "'=2+2"
    assert safe_csv_cell("  @cmd") == "'  @cmd"
    assert safe_csv_cell("normal") == "normal"
    assert safe_csv([]) == b""
    exported = safe_csv([{"title": "=formula", "count": 2}]).decode("utf-8-sig")
    assert "'=formula" in exported


def test_knowledge_client_contract_and_transient_headers(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def request(method: str, url: str, **kwargs: object) -> Response:
        calls.append({"method": method, "url": url, **kwargs})
        if url.endswith("/v1/versions"):
            return Response([{"version": "v1"}])
        return Response({"batch_id": "b1", "job_id": "j1", "version": "v1"})

    monkeypatch.setattr(httpx, "request", request)
    client = KnowledgeService(
        "http://service/",
        "service-secret",
        "api-secret",
        "https://api.test/v1",
        "environment-admin",
    )
    assert client.create_batch("session", "../papers.csv", [{"title": "A"}])["batch_id"] == "b1"
    assert client.get_batch("b1")["batch_id"] == "b1"
    assert client.approve("b1", "session-admin")["job_id"] == "j1"
    assert client.get_job("j1")["job_id"] == "j1"
    assert client.versions()[0]["version"] == "v1"
    assert client.activate("v1", "session-admin")["version"] == "v1"
    assert client.rollback("session-admin")["version"] == "v1"
    create_call = calls[0]
    assert create_call["json"]["source_file"] == "papers.csv"
    assert create_call["headers"]["X-ENRRCREW-UPSTREAM-KEY"] == "api-secret"
    approve_call = next(call for call in calls if call["url"].endswith("/approve"))
    assert approve_call["headers"]["X-ENRRCREW-ADMIN-TOKEN"] == "session-admin"


def test_knowledge_client_errors_are_sanitized(monkeypatch) -> None:
    client = KnowledgeService("http://service", "token", "secret", "https://api.test")
    monkeypatch.setattr(
        httpx,
        "request",
        lambda *args, **kwargs: Response({"detail": "Safe rejection"}, 409),
    )
    with pytest.raises(RuntimeError, match="Safe rejection"):
        client.get_batch("bad")
    monkeypatch.setattr(
        httpx,
        "request",
        lambda *args, **kwargs: Response({}, 500, invalid_json=True),
    )
    with pytest.raises(RuntimeError, match="rejected"):
        client.get_batch("bad")

    def offline(*args: object, **kwargs: object):
        raise httpx.ConnectError("secret D:\\private")

    monkeypatch.setattr(httpx, "request", offline)
    with pytest.raises(RuntimeError, match="unavailable") as caught:
        client.get_batch("bad")
    assert "secret" not in str(caught.value)


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class FakeColumn:
    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.events = events

    def metric(self, label: str, value: object) -> None:
        self.events.append((label, value))


class FakeStreamlit:
    def __init__(self, state: dict[str, object]) -> None:
        self.session_state = AttrDict(state)
        self.events: list[tuple[str, object]] = []

    def progress(self, value: int, *, text: str = "") -> None:
        self.events.append(("progress", (value, text)))

    def columns(self, count: int) -> list[FakeColumn]:
        return [FakeColumn(self.events) for _ in range(count)]

    def caption(self, value: str) -> None:
        self.events.append(("caption", value))

    def info(self, value: str) -> None:
        self.events.append(("info", value))

    def markdown(self, value: str) -> None:
        self.events.append(("markdown", value))

    def expander(self, *args: object, **kwargs: object):
        self.events.append(("expander", args[0]))
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def error(self, value: str) -> None:
        self.events.append(("error", value))

    def warning(self, value: str) -> None:
        self.events.append(("warning", value))

    def success(self, value: str) -> None:
        self.events.append(("success", value))

    def button(self, *args: object, **kwargs: object) -> bool:
        return False

    def rerun(self, **kwargs: object) -> None:
        self.events.append(("rerun", kwargs))


def test_job_progress_helpers_and_auto_refresh_panel(monkeypatch) -> None:
    heartbeat = (datetime.now(UTC) - timedelta(seconds=2)).isoformat()
    active = {
        "job_id": "job-one",
        "batch_id": "batch-one",
        "status": "indexing",
        "progress": 50,
        "message": "Extracting entities",
        "elapsed_seconds": 65,
        "heartbeat_at": heartbeat,
    }
    published = {
        **active,
        "status": "published",
        "progress": 100,
        "message": "Index release published and activated",
        "version": "v-new",
    }
    fake_st = FakeStreamlit(
        {"kb_job": active, "kb_batch": {"batch_id": "batch-one"}}
    )
    client = SimpleNamespace(
        get_job=lambda job_id: published,
        get_batch=lambda batch_id: {"batch_id": batch_id, "status": "published"},
    )
    monkeypatch.setattr(knowledge_ui, "st", fake_st)
    monkeypatch.setattr(knowledge_ui, "_client", lambda settings: client)

    assert knowledge_ui._job_is_active(active) is True
    assert knowledge_ui._job_is_active(published) is False
    assert knowledge_ui._format_elapsed(65) == "1m 05s"
    assert knowledge_ui._format_elapsed(3661) == "1h 01m 01s"
    assert knowledge_ui._seconds_since("invalid") is None
    knowledge_ui._render_job_panel(SimpleNamespace())

    assert fake_st.session_state.kb_job["status"] == "published"
    assert fake_st.session_state.kb_batch["status"] == "published"
    assert ("progress", (100, "published: Index release published and activated")) in fake_st.events
    assert ("Stage", "published") in fake_st.events
    assert any(event[0] == "success" for event in fake_st.events)


def test_knowledge_tutorial_recommends_the_next_safe_action(monkeypatch) -> None:
    fake_st = FakeStreamlit({})
    monkeypatch.setattr(knowledge_ui, "st", fake_st)
    assert (
        knowledge_ui._knowledge_next_action(None, None)[0]
        == "Upload and validate a literature table"
    )
    assert knowledge_ui._knowledge_next_action(
        {"status": "needs_review", "accepted": 1}, None
    )[0] == "Review and approve the batch"
    assert knowledge_ui._knowledge_next_action(
        {"status": "validated", "accepted": 0}, None
    )[0] == "No new index is required"
    assert knowledge_ui._knowledge_next_action(
        None, {"status": "indexing"}
    )[0] == "Wait for indexing to finish"
    assert knowledge_ui._knowledge_next_action(
        None, {"status": "published"}
    )[0] == "Verify the new version"
    assert knowledge_ui._knowledge_next_action(
        None, {"status": "interrupted"}
    )[0] == "Review the failure before retrying"

    knowledge_ui._render_knowledge_tutorial(
        {"status": "needs_review", "accepted": 1}, None
    )
    assert any(
        event[0] == "info" and "Review and approve the batch" in str(event[1])
        for event in fake_st.events
    )
    assert any(
        event[0] == "markdown" and "Activate selected version" in str(event[1])
        for event in fake_st.events
    )
