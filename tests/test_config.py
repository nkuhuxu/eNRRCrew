from pathlib import Path

import pytest

from enrrcrew.config import AppSettings, SessionWorkspace


def settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        project_root=tmp_path,
        asset_root=tmp_path / "assets",
        runtime_root=tmp_path / "runtime",
        knowledge_root=tmp_path / "knowledge",
        rag_service_url="http://127.0.0.1:8765",
        rag_service_token="service-token",
        admin_token="admin-token",
        api_key="environment-key",
        base_url="https://example.test/v1",
        chat_model="chat",
        manager_model="manager",
        agent_max_rounds=8,
        agent_timeout=180,
        sandbox_image="sandbox",
        sandbox_timeout=30,
    )


def test_session_workspaces_are_isolated(tmp_path: Path) -> None:
    first = SessionWorkspace.create(settings(tmp_path), "session-one")
    second = SessionWorkspace.create(settings(tmp_path), "session-two")
    assert first.root != second.root
    assert first.output_dir.is_dir()
    assert second.output_dir.is_dir()
    (first.root / "private.txt").write_text("first", encoding="utf-8")
    (second.root / "private.txt").write_text("second", encoding="utf-8")
    assert (first.root / "private.txt").read_text(encoding="utf-8") == "first"
    assert (second.root / "private.txt").read_text(encoding="utf-8") == "second"


def test_session_identifier_cannot_escape_runtime_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="letters, numbers, and hyphens"):
        SessionWorkspace.create(settings(tmp_path), "../other-session")


def test_default_configuration_uses_only_repository_assets(monkeypatch) -> None:
    monkeypatch.delenv("ENRRCREW_ASSET_ROOT", raising=False)
    monkeypatch.delenv("ENRRCREW_RUNTIME_ROOT", raising=False)
    configured = AppSettings.from_environment()
    repository_root = Path(__file__).resolve().parents[1]

    assert configured.asset_root == repository_root
    assert configured.runtime_root == repository_root / "runtime"
    assert configured.knowledge_root == repository_root / "knowledge"
    assert configured.validate_assets() == []
    assert (configured.asset_root / "prompts").is_dir()
    assert configured.rag_service_url == "http://127.0.0.1:8765"
    assert configured.agent_max_rounds == 8
    assert configured.agent_timeout == 180


def test_agent_limits_are_loaded_and_bounded(monkeypatch) -> None:
    monkeypatch.setenv("ENRRCREW_AGENT_MAX_ROUNDS", "1")
    monkeypatch.setenv("ENRRCREW_AGENT_TIMEOUT", "0")

    configured = AppSettings.from_environment()

    assert configured.agent_max_rounds == 2
    assert configured.agent_timeout == 1


def test_session_credentials_override_without_mutating_base(tmp_path: Path) -> None:
    original = settings(tmp_path)
    changed = original.with_session_credentials("temporary", "https://override.test/v1")
    assert original.api_key == "environment-key"
    assert changed.api_key == "temporary"
    assert changed.base_url == "https://override.test/v1"
