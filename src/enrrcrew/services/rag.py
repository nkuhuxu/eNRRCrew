from __future__ import annotations

from typing import Literal

import httpx
from pydantic import BaseModel, Field


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


class RagService:
    """Compatibility-preserving client for the isolated GraphRAG service."""

    def __init__(
        self,
        service_url: str,
        service_token: str,
        api_key: str,
        upstream_base_url: str,
        timeout: float = 180.0,
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.service_token = service_token
        self.api_key = api_key
        self.upstream_base_url = upstream_base_url
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"X-ENRRCREW-SERVICE-TOKEN": self.service_token}
        if self.api_key:
            headers["X-ENRRCREW-UPSTREAM-KEY"] = self.api_key
        if self.upstream_base_url:
            headers["X-ENRRCREW-UPSTREAM-BASE-URL"] = self.upstream_base_url
        return headers

    def search_result(
        self,
        question: str,
        mode: Literal["local", "global"] = "local",
        community: int = 0,
        response_type: str = "single paragraph",
    ) -> RagQueryResult:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if not self.api_key:
            raise ValueError("An API key is required for GraphRAG search")
        try:
            response = httpx.post(
                f"{self.service_url}/v1/query",
                headers=self._headers(),
                json={
                    "question": question,
                    "mode": mode,
                    "community_level": community,
                    "response_type": response_type,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError("GraphRAG service is unavailable or rejected the query") from exc
        return RagQueryResult.model_validate(response.json())

    def search(
        self,
        question: str,
        mode: Literal["local", "global"] = "local",
        community: int = 0,
        response_type: str = "single paragraph",
    ) -> str:
        return self.search_result(question, mode, community, response_type).answer

    def health(self) -> dict[str, object]:
        try:
            response = httpx.get(f"{self.service_url}/health", timeout=3.0)
            response.raise_for_status()
            return dict(response.json())
        except httpx.HTTPError:
            return {"status": "unavailable"}
