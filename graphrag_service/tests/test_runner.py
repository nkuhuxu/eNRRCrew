from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from enrrcrew_rag.corpus import make_document
from enrrcrew_rag.runner import PROMPT_GUARD, GraphRunner, graph_settings, indexing_progress
from enrrcrew_rag.schemas import BatchDocument, RagQueryRequest
from enrrcrew_rag.store import ControlStore, read_active


def document(name: str = "one"):
    return make_document(
        BatchDocument(
            title=f"Catalyst {name}",
            abstract=(f"Nitrogen reduction evidence for catalyst {name}. " * 5),
            doi=f"10.test/{name}",
            publication_year=2025,
        ),
        "source.csv",
    )


class FakeCommands:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail = False

    def __call__(self, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        if self.fail:
            return subprocess.CompletedProcess(args, 1, "", "secret internal failure")
        root = Path(kwargs["cwd"])
        if "query" not in args:
            target = root / ("update_output" if "update" in args else "output")
            target.mkdir(parents=True, exist_ok=True)
            if target.name == "update_output":
                target = root / "output"
            rows = sum(
                1
                for line in (root / "input" / "documents.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            )
            pd.DataFrame({"id": list(range(rows))}).to_parquet(
                target / "documents.parquet", index=False
            )
            return subprocess.CompletedProcess(args, 0, "indexed", "")
        return subprocess.CompletedProcess(
            args, 0, "Catalyst one is discussed (10.test/one).", ""
        )


def test_settings_are_current_and_do_not_persist_endpoint() -> None:
    config = graph_settings()
    assert "model_provider: openai" in config
    assert "model: text-embedding-3-small" in config
    assert "vector_size: 1536" in config
    assert "type: jsonl" in config
    assert "${GRAPHRAG_BASE_URL}" in config
    assert "https://" not in config


def test_build_update_query_and_failure_leave_active_stable(tmp_path: Path) -> None:
    store = ControlStore(tmp_path / "state" / "control.sqlite", tmp_path)
    commands = FakeCommands()
    runner = GraphRunner(tmp_path, store, commands)
    progress: list[tuple[str, int, str]] = []
    with store.writer():
        first = runner.build_release(
            [document()],
            "secret-key",
            "https://api.test/v1",
            update=False,
            progress_callback=lambda stage, value, message: progress.append(
                (stage, value, message)
            ),
        )
    assert [(stage, value) for stage, value, _ in progress] == [
        ("preparing", 5),
        ("preparing", 10),
        ("preparing", 15),
        ("indexing", 20),
        ("validating", 95),
        ("publishing", 98),
    ]
    active = read_active(tmp_path / "active.json")
    assert active["version"] == first
    release = tmp_path / "releases" / first
    assert PROMPT_GUARD in (release / "prompts" / "extract_graph.txt").read_text(
        encoding="utf-8"
    )
    assert "secret-key" not in (release / "settings.yaml").read_text(encoding="utf-8")
    assert "api.test" not in (release / "settings.yaml").read_text(encoding="utf-8")

    with store.writer():
        second = runner.build_release(
            [document(), document("two")], "secret-key", "https://api.test/v1", update=True
        )
    assert second != first
    assert read_active(tmp_path / "active.json")["previous_version"] == first
    assert any("update" in call for call in commands.calls)
    assert not any("standard-update" in call for call in commands.calls)

    runner._execute_query = lambda *args: (
        "Catalyst one is discussed (10.test/one).",
        {"sources": pd.DataFrame({"document_id": ["doi:10.test/one"]})},
    )
    result = runner.query(
        RagQueryRequest(question="What about catalyst one?", community_level=1),
        "secret-key",
        "https://api.test/v1",
    )
    assert result.index_version == second
    assert result.citations[0].doi == "10.test/one"
    assert result.citations[0].document_id == "doi:10.test/one"

    commands.fail = True
    before = read_active(tmp_path / "active.json")
    with pytest.raises(RuntimeError, match="indexing failed"):
        runner.build_release([document("three")], "secret", "https://api.test", update=True)
    assert read_active(tmp_path / "active.json") == before
    runner._execute_query = lambda *args: (_ for _ in ()).throw(
        RuntimeError("query failed")
    )
    with pytest.raises(RuntimeError, match="query failed"):
        runner.query(RagQueryRequest(question="test"), "secret", "https://api.test")


def test_runner_requires_credentials_and_active_index(tmp_path: Path) -> None:
    store = ControlStore(tmp_path / "control.sqlite", tmp_path)
    runner = GraphRunner(tmp_path, store, FakeCommands())
    with pytest.raises(ValueError, match="API key"):
        runner.build_release([document()], "", "https://api.test", update=False)
    with pytest.raises(RuntimeError, match="No active"):
        runner.query(RagQueryRequest(question="test"), "key", "https://api.test")


def test_indexing_progress_uses_only_artifacts_changed_by_current_run(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    marker = 2_000_000_000.0
    documents = output / "documents.parquet"
    documents.write_text("old", encoding="utf-8")
    documents.touch()
    documents_time = marker - 100
    import os

    os.utime(documents, (documents_time, documents_time))
    assert indexing_progress(tmp_path, marker)[0] == 20
    entities = output / "entities.parquet"
    entities.write_text("new", encoding="utf-8")
    os.utime(entities, (marker + 1, marker + 1))
    assert indexing_progress(tmp_path, marker) == (
        50,
        "Extracting entities and relationships",
    )
    lancedb = output / "lancedb"
    lancedb.mkdir()
    vector = lancedb / "data.lance"
    vector.write_text("new", encoding="utf-8")
    os.utime(vector, (marker + 2, marker + 2))
    assert indexing_progress(tmp_path, marker)[0] == 90


def test_in_process_query_config_and_local_global_execution(monkeypatch, tmp_path: Path) -> None:
    import graphrag.api as api

    store = ControlStore(tmp_path / "control.sqlite", tmp_path)
    runner = GraphRunner(tmp_path, store, FakeCommands())
    root = tmp_path / "release"
    (root / "prompts").mkdir(parents=True)
    config = runner._query_config(
        root, "output", "transient-key", "https://endpoint.test/v1"
    )
    assert config.vector_store.vector_size == 1536
    assert config.completion_models["default_completion_model"].api_key == "transient-key"
    assert config.output_storage.base_dir == str((root / "output").resolve())

    tables = {
        "entities": pd.DataFrame(),
        "communities": pd.DataFrame(),
        "community_reports": pd.DataFrame(),
        "text_units": pd.DataFrame(),
        "relationships": pd.DataFrame(),
        "covariates": None,
    }
    monkeypatch.setattr(runner, "_query_config", lambda *args: SimpleNamespace())
    monkeypatch.setattr(runner, "_tables_for", lambda *args: tables)

    async def local_search(**kwargs):
        return "local answer", {"sources": pd.DataFrame()}

    async def global_search(**kwargs):
        return {"answer": "global"}, {}

    monkeypatch.setattr(api, "local_search", local_search)
    monkeypatch.setattr(api, "global_search", global_search)
    local, _ = runner._execute_query(
        root,
        "v1",
        "output",
        RagQueryRequest(question="local", mode="local"),
        "key",
        "https://api.test",
    )
    global_answer, _ = runner._execute_query(
        root,
        "v1",
        "output",
        RagQueryRequest(question="global", mode="global"),
        "key",
        "https://api.test",
    )
    assert local == "local answer"
    assert global_answer == '{"answer": "global"}'
