from __future__ import annotations

import httpx
import pytest

from enrrcrew.services.rag import RagService


class Response:
    def __init__(self, payload: dict[str, object], status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://127.0.0.1/v1/query")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("failed", request=request, response=response)

    def json(self) -> dict[str, object]:
        return self.payload


def test_rag_search_returns_structured_result_and_compatibility_text(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def post(url: str, **kwargs: object) -> Response:
        calls.append({"url": url, **kwargs})
        return Response(
            {
                "answer": "evidence",
                "mode": "local",
                "index_version": "v1",
                "citations": [
                    {
                        "document_id": "doi:10.test/example",
                        "title": "NRR evidence",
                        "doi": "10.test/example",
                    }
                ],
                "elapsed_ms": 12,
            }
        )

    monkeypatch.setattr(httpx, "post", post)
    service = RagService(
        "http://127.0.0.1:8765", "service-token", "session-key", "https://api.test/v1"
    )
    result = service.search_result("question", "local", 2, "prioritized list")
    assert result.answer == "evidence"
    assert result.citations[0].doi == "10.test/example"
    assert service.search("question") == "evidence"
    assert calls[0]["url"] == "http://127.0.0.1:8765/v1/query"
    headers = calls[0]["headers"]
    assert headers["X-ENRRCREW-SERVICE-TOKEN"] == "service-token"
    assert headers["X-ENRRCREW-UPSTREAM-KEY"] == "session-key"
    assert calls[0]["json"]["community_level"] == 2


def test_rag_search_validates_input_and_redacts_transport_failures(monkeypatch) -> None:
    service = RagService("http://service", "token", "", "https://api.test/v1")
    with pytest.raises(ValueError, match="API key"):
        service.search("question")
    service.api_key = "secret"
    with pytest.raises(ValueError, match="Question"):
        service.search(" ")
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: Response({}, 503))
    with pytest.raises(RuntimeError, match="unavailable") as caught:
        service.search("question")
    assert "secret" not in str(caught.value)


def test_rag_health_is_non_throwing(monkeypatch) -> None:
    service = RagService("http://service", "token", "key", "https://api.test/v1")
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: Response({"status": "ok"}))
    assert service.health()["status"] == "ok"

    def fail(*args: object, **kwargs: object) -> Response:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", fail)
    assert service.health() == {"status": "unavailable"}
