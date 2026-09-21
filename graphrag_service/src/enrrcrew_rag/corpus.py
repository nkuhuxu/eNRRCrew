from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import BatchDocument, CorpusDocument

WHITESPACE = re.compile(r"\s+")
DOI_PREFIX = re.compile(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)", re.IGNORECASE)


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return WHITESPACE.sub(" ", unicodedata.normalize("NFKC", str(value))).strip()


def normalize_doi(value: object) -> str | None:
    text = normalize_text(value).lower()
    text = DOI_PREFIX.sub("", text).strip()
    return text or None


def content_hash(title: str, abstract: str) -> str:
    normalized = f"{normalize_text(title)}\n{normalize_text(abstract)}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_document(
    value: BatchDocument,
    source_file: str,
    *,
    source_sheet: str | None = None,
    source_row: int | None = None,
) -> CorpusDocument:
    title = normalize_text(value.title)
    abstract = normalize_text(value.abstract)
    doi = normalize_doi(value.doi)
    digest = content_hash(title, abstract)
    return CorpusDocument(
        document_id=f"doi:{doi}" if doi else f"sha256:{digest}",
        title=title,
        abstract=abstract,
        publication_year=value.publication_year,
        doi=doi,
        content_sha256=digest,
        source_file=Path(source_file).name,
        source_sheet=source_sheet,
        source_row=source_row if source_row is not None else value.source_row,
    )


def read_jsonl(path: Path) -> list[CorpusDocument]:
    if not path.exists():
        return []
    return [
        CorpusDocument.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, documents: list[CorpusDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(documents, key=lambda item: item.document_id)
    payload = "".join(
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
        for item in ordered
    )
    path.write_text(payload, encoding="utf-8", newline="\n")


def merge_documents(
    current: list[CorpusDocument], incoming: list[CorpusDocument]
) -> tuple[list[CorpusDocument], list[dict[str, Any]], dict[str, int]]:
    by_id = {item.document_id: item for item in current}
    by_hash = {item.content_sha256: item for item in current}
    issues: list[dict[str, Any]] = []
    counts = {"accepted": 0, "duplicates": 0, "conflicts": 0}
    for item in incoming:
        existing_id = by_id.get(item.document_id)
        existing_hash = by_hash.get(item.content_sha256)
        if existing_hash or (existing_id and existing_id.content_sha256 == item.content_sha256):
            counts["duplicates"] += 1
            issues.append({"source_row": item.source_row, "kind": "duplicate"})
            continue
        if existing_id:
            counts["conflicts"] += 1
            issues.append(
                {
                    "source_row": item.source_row,
                    "kind": "doi_conflict",
                    "document_id": item.document_id,
                }
            )
            continue
        by_id[item.document_id] = item
        by_hash[item.content_sha256] = item
        counts["accepted"] += 1
    return sorted(by_id.values(), key=lambda item: item.document_id), issues, counts


def migrate_baseline(raw_path: Path, merged_path: Path, output_path: Path) -> dict[str, Any]:
    frame = pd.read_excel(raw_path, sheet_name="savedrecs", engine="xlrd")
    abstract_column = "Abstract"
    title_column = "Article Title"
    year_column = "Publication Year"
    doi_column = "DOI"
    normalized_rows: dict[str, list[int]] = {}
    for index, abstract in frame[abstract_column].items():
        normalized_rows.setdefault(normalize_text(abstract), []).append(int(index))
    merged = [line for line in merged_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matched: list[CorpusDocument] = []
    for abstract in merged:
        indexes = normalized_rows.get(normalize_text(abstract), [])
        if len(indexes) != 1:
            raise ValueError(f"Expected one source match, found {len(indexes)}")
        index = indexes[0]
        row = frame.loc[index]
        year_value = row.get(year_column)
        year = None if pd.isna(year_value) else int(year_value)
        matched.append(
            make_document(
                BatchDocument(
                    title=normalize_text(row.get(title_column)),
                    abstract=normalize_text(row.get(abstract_column)),
                    publication_year=year,
                    doi=normalize_doi(row.get(doi_column)),
                ),
                raw_path.name,
                source_sheet="savedrecs",
                source_row=index + 2,
            )
        )
    by_id: dict[str, CorpusDocument] = {}
    duplicate_audit: list[dict[str, Any]] = []
    for document in matched:
        previous = by_id.get(document.document_id)
        if previous is None:
            by_id[document.document_id] = document
            continue
        keep, skip = sorted(
            (previous, document),
            key=lambda item: (-len(item.abstract), item.source_row or 0),
        )
        by_id[document.document_id] = keep
        duplicate_audit.append(
            {
                "document_id": document.document_id,
                "kept_source_row": keep.source_row,
                "skipped_source_row": skip.source_row,
                "reason": "duplicate DOI; retained the longer normalized abstract",
            }
        )
    documents = sorted(by_id.values(), key=lambda item: item.document_id)
    write_jsonl(output_path, documents)
    audit = {
        "source_rows": len(frame),
        "merged_abstracts": len(merged),
        "matched_rows": len(matched),
        "baseline_documents": len(documents),
        "excluded_unique_documents": 1819,
        "duplicates": duplicate_audit,
    }
    audit_path = output_path.with_name("migration_audit.json")
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return audit
