from __future__ import annotations

import csv
import io
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UPLOAD_ROWS = 1000

COLUMN_ALIASES = {
    "title": {"title", "article title", "论文标题", "标题"},
    "abstract": {"abstract", "摘要", "summary"},
    "doi": {"doi", "digital object identifier"},
    "publication_year": {"publication year", "year", "年份", "发表年份"},
}


def detect_columns(columns: list[str]) -> dict[str, str | None]:
    normalized = {str(column).strip().casefold(): str(column) for column in columns}
    return {
        field: next(
            (normalized[alias] for alias in aliases if alias in normalized), None
        )
        for field, aliases in COLUMN_ALIASES.items()
    }


def read_upload(filename: str, payload: bytes) -> pd.DataFrame:
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload exceeds the 20 MB limit")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".xls", ".xlsx"}:
        raise ValueError("Only CSV, XLS, and XLSX files are accepted")
    stream = io.BytesIO(payload)
    if suffix == ".csv":
        frame = pd.read_csv(stream)
    elif suffix == ".xls":
        frame = pd.read_excel(stream, engine="xlrd")
    else:
        frame = pd.read_excel(stream, engine="openpyxl")
    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError("Upload exceeds the 1,000-document limit")
    return frame


def frame_to_documents(
    frame: pd.DataFrame, mapping: dict[str, str | None]
) -> list[dict[str, object]]:
    title_column = mapping.get("title")
    abstract_column = mapping.get("abstract")
    if not title_column or not abstract_column:
        raise ValueError("Title and abstract columns are required")
    documents: list[dict[str, object]] = []
    for index, row in frame.iterrows():
        year: int | None = None
        year_column = mapping.get("publication_year")
        if year_column and pd.notna(row.get(year_column)):
            try:
                year = int(row[year_column])
            except (TypeError, ValueError):
                year = -1
        doi_column = mapping.get("doi")
        doi = None
        if doi_column and pd.notna(row.get(doi_column)):
            doi = str(row[doi_column])
        documents.append(
            {
                "title": "" if pd.isna(row.get(title_column)) else str(row[title_column]),
                "abstract": (
                    "" if pd.isna(row.get(abstract_column)) else str(row[abstract_column])
                ),
                "doi": doi,
                "publication_year": year,
                "source_row": int(index) + 2,
            }
        )
    return documents


def safe_csv_cell(value: object) -> object:
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def safe_csv(rows: list[dict[str, object]]) -> bytes:
    if not rows:
        return b""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(
        {key: safe_csv_cell(value) for key, value in row.items()} for row in rows
    )
    return output.getvalue().encode("utf-8-sig")


class KnowledgeService:
    def __init__(
        self,
        base_url: str,
        service_token: str,
        api_key: str,
        upstream_base_url: str,
        admin_token: str = "",
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.service_token = service_token
        self.api_key = api_key
        self.upstream_base_url = upstream_base_url
        self.admin_token = admin_token
        self.timeout = timeout

    def _headers(self, admin_token: str = "") -> dict[str, str]:
        headers = {
            "X-ENRRCREW-SERVICE-TOKEN": self.service_token,
            "X-ENRRCREW-UPSTREAM-KEY": self.api_key,
            "X-ENRRCREW-UPSTREAM-BASE-URL": self.upstream_base_url,
        }
        selected_admin = admin_token or self.admin_token
        if selected_admin:
            headers["X-ENRRCREW-ADMIN-TOKEN"] = selected_admin
        return headers

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers(kwargs.pop("admin_token", "")),
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = "Request rejected by the GraphRAG service"
            with suppress(ValueError):
                detail = str(exc.response.json().get("detail", detail))
            raise RuntimeError(detail) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError("GraphRAG service is unavailable") from exc
        return response.json()

    def create_batch(
        self,
        session_id: str,
        source_file: str,
        documents: list[dict[str, object]],
    ) -> dict[str, object]:
        return dict(
            self._request(
                "POST",
                "/v1/batches",
                json={
                    "session_id": session_id,
                    "source_file": Path(source_file).name,
                    "documents": documents,
                },
            )
        )

    def get_batch(self, batch_id: str) -> dict[str, object]:
        return dict(self._request("GET", f"/v1/batches/{batch_id}"))

    def approve(self, batch_id: str, admin_token: str) -> dict[str, object]:
        return dict(
            self._request(
                "POST", f"/v1/batches/{batch_id}/approve", admin_token=admin_token
            )
        )

    def get_job(self, job_id: str) -> dict[str, object]:
        return dict(self._request("GET", f"/v1/jobs/{job_id}"))

    def versions(self) -> list[dict[str, object]]:
        return list(self._request("GET", "/v1/versions"))

    def activate(self, version: str, admin_token: str) -> dict[str, object]:
        return dict(
            self._request(
                "POST", f"/v1/versions/{version}/activate", admin_token=admin_token
            )
        )

    def rollback(self, admin_token: str) -> dict[str, object]:
        return dict(
            self._request("POST", "/v1/versions/rollback", admin_token=admin_token)
        )
