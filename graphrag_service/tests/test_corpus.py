from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from enrrcrew_rag.corpus import (
    content_hash,
    make_document,
    merge_documents,
    migrate_baseline,
    normalize_doi,
    normalize_text,
    read_jsonl,
    write_jsonl,
)
from enrrcrew_rag.schemas import BatchDocument


def raw(value: str, doi: str | None = None, row: int = 2) -> BatchDocument:
    return BatchDocument(
        title=f"Title {value}",
        abstract=(f"Abstract {value} " * 10).strip(),
        doi=doi,
        publication_year=2024,
        source_row=row,
    )


def test_normalization_hash_and_document_identifiers() -> None:
    assert normalize_text("  A\u3000B\n C ") == "A B C"
    assert normalize_doi("HTTPS://DOI.ORG/10.1000/ABC ") == "10.1000/abc"
    assert normalize_doi(None) is None
    digest = content_hash(" A ", "B")
    assert digest == hashlib.sha256(b"A\nB").hexdigest()
    with_doi = make_document(raw("one", "doi:10.1/ONE"), "../upload.csv")
    without_doi = make_document(raw("two"), "upload.csv")
    assert with_doi.document_id == "doi:10.1/one"
    assert with_doi.source_file == "upload.csv"
    assert without_doi.document_id.startswith("sha256:")


def test_jsonl_is_byte_deterministic_and_merge_classifies(tmp_path: Path) -> None:
    first = make_document(raw("first", "10.1/first"), "a.csv")
    second = make_document(raw("second"), "a.csv")
    path = tmp_path / "documents.jsonl"
    write_jsonl(path, [second, first])
    initial = path.read_bytes()
    write_jsonl(path, [first, second])
    assert path.read_bytes() == initial
    assert [item.document_id for item in read_jsonl(path)] == sorted(
        [first.document_id, second.document_id]
    )

    duplicate = make_document(raw("first", "10.1/first", 3), "b.csv")
    conflict = make_document(raw("changed", "10.1/first", 4), "b.csv")
    new = make_document(raw("new", "10.1/new", 5), "b.csv")
    merged, issues, counts = merge_documents([first], [duplicate, conflict, new])
    assert len(merged) == 2
    assert counts == {"accepted": 1, "duplicates": 1, "conflicts": 1}
    assert {item["kind"] for item in issues} == {"duplicate", "doi_conflict"}


def test_schema_rejects_invalid_length_and_year() -> None:
    with pytest.raises(ValueError):
        make_document(
            BatchDocument(title="x", abstract="short", publication_year=1700), "x.csv"
        )


def test_real_baseline_migration_matches_exact_contract(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[2]
    source = project / "knowledge" / "source"
    output = tmp_path / "documents.jsonl"
    audit = migrate_baseline(
        source / "raw_corpus.xls", source / "merged_abstracts.txt", output
    )
    assert audit["matched_rows"] == 500
    assert audit["baseline_documents"] == 499
    assert audit["excluded_unique_documents"] == 1819
    assert audit["duplicates"][0]["document_id"] == "doi:10.1021/acsaem.3c01382"
    assert len(read_jsonl(output)) == 499
    second = tmp_path / "second" / "documents.jsonl"
    migrate_baseline(source / "raw_corpus.xls", source / "merged_abstracts.txt", second)
    assert output.read_bytes() == second.read_bytes()
