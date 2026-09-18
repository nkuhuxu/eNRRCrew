from pathlib import Path

import pytest

from enrrcrew.config import AppSettings, SessionWorkspace


def settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        project_root=tmp_path,
        asset_root=tmp_path / "assets",
        runtime_root=tmp_path / "runtime",
        api_key="environment-key",
        base_url="https://example.test/v1",
        chat_model="chat",
        manager_model="manager",
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
    repository_root = Path(__file__).resolve().parents[2]

    assert configured.asset_root == repository_root
    assert configured.runtime_root == repository_root / "uv" / "runtime"
    assert configured.validate_assets() == []
    assert (configured.asset_root / "output").is_dir()
    assert (configured.asset_root / "prompts").is_dir()


def test_session_credentials_override_without_mutating_base(tmp_path: Path) -> None:
    original = settings(tmp_path)
    changed = original.with_session_credentials("temporary", "https://override.test/v1")
    assert original.api_key == "environment-key"
    assert changed.api_key == "temporary"
    assert changed.base_url == "https://override.test/v1"
