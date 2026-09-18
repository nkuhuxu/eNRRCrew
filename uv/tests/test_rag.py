import os
from pathlib import Path

import pytest

from enrrcrew.services.rag import RagService, _temporary_graphrag_environment


def test_temporary_rag_credentials_are_restored(monkeypatch) -> None:
    monkeypatch.setenv("GRAPHRAG_API_KEY", "original-key")
    monkeypatch.setenv("GRAPHRAG_BASE_URL", "https://original.test/v1")
    with _temporary_graphrag_environment("session-key", "https://session.test/v1"):
        assert os.environ["GRAPHRAG_API_KEY"] == "session-key"
        assert os.environ["GRAPHRAG_BASE_URL"] == "https://session.test/v1"
    assert os.environ["GRAPHRAG_API_KEY"] == "original-key"
    assert os.environ["GRAPHRAG_BASE_URL"] == "https://original.test/v1"


@pytest.mark.parametrize(
    ("mode", "runner_name"),
    [("local", "run_local_search"), ("global", "run_global_search")],
)
def test_rag_search_forwards_query_options_and_restores_environment(
    monkeypatch, tmp_path: Path, mode: str, runner_name: str
) -> None:
    from graphrag.query import cli

    config = tmp_path / "settings.yaml"
    config.write_text("encoding_model: cl100k_base\n", encoding="utf-8")
    calls: list[tuple[object, ...]] = []

    def fake_runner(*args: object) -> tuple[str, dict[str, object]]:
        calls.append(args)
        assert os.environ["GRAPHRAG_API_KEY"] == "session-key"
        assert os.environ["GRAPHRAG_BASE_URL"] == "https://session.test/v1"
        return "mock answer", {}

    monkeypatch.setattr(cli, runner_name, fake_runner)
    monkeypatch.setenv("GRAPHRAG_API_KEY", "host-key")
    monkeypatch.setenv("GRAPHRAG_BASE_URL", "https://host.test/v1")
    service = RagService(
        tmp_path, config, "session-key", "https://session.test/v1"
    )

    result = service.search("What controls selectivity?", mode, 2, "prioritized list")

    assert result == "mock answer"
    assert calls == [
        (
            str(config.resolve()),
            None,
            str(tmp_path.resolve()),
            2,
            "prioritized list",
            True,
            "What controls selectivity?",
        )
    ]
    assert os.environ["GRAPHRAG_API_KEY"] == "host-key"
    assert os.environ["GRAPHRAG_BASE_URL"] == "https://host.test/v1"


def test_rag_search_fails_safely_without_credentials(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("test", encoding="utf-8")
    service = RagService(tmp_path, config, "", "https://test/v1")
    with pytest.raises(ValueError, match="API key"):
        service.search("question")
