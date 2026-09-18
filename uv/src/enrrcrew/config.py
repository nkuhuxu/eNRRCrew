from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@dataclass(frozen=True, slots=True)
class AppSettings:
    project_root: Path
    asset_root: Path
    runtime_root: Path
    api_key: str
    base_url: str
    chat_model: str
    manager_model: str
    sandbox_image: str
    sandbox_timeout: int

    @classmethod
    def from_environment(cls) -> AppSettings:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        asset_root = _resolve_path(
            os.getenv("ENRRCREW_ASSET_ROOT", ".."), PROJECT_ROOT
        )
        runtime_root = _resolve_path(
            os.getenv("ENRRCREW_RUNTIME_ROOT", "./runtime"), PROJECT_ROOT
        )
        return cls(
            project_root=PROJECT_ROOT,
            asset_root=asset_root,
            runtime_root=runtime_root,
            api_key=os.getenv("GRAPHRAG_API_KEY", ""),
            base_url=os.getenv("GRAPHRAG_BASE_URL", "https://api.openai.com/v1"),
            chat_model=os.getenv("ENRRCREW_CHAT_MODEL", "gpt-4o-mini"),
            manager_model=os.getenv("ENRRCREW_MANAGER_MODEL", "gpt-4o"),
            sandbox_image=os.getenv(
                "ENRRCREW_SANDBOX_IMAGE", "enrrcrew-csv-sandbox:local"
            ),
            sandbox_timeout=int(os.getenv("ENRRCREW_SANDBOX_TIMEOUT", "30")),
        )

    def with_session_credentials(self, api_key: str, base_url: str) -> AppSettings:
        return replace(
            self,
            api_key=api_key.strip() or self.api_key,
            base_url=base_url.strip() or self.base_url,
        )

    @property
    def models_dir(self) -> Path:
        return self.asset_root / "models"

    @property
    def input_dir(self) -> Path:
        return self.asset_root / "input"

    @property
    def graphrag_config(self) -> Path:
        return self.asset_root / "settings.yaml"

    def validate_assets(self) -> list[str]:
        required = [self.models_dir, self.input_dir, self.graphrag_config]
        return [str(path) for path in required if not path.exists()]


@dataclass(frozen=True, slots=True)
class SessionWorkspace:
    session_id: str
    root: Path
    output_dir: Path

    @classmethod
    def create(cls, settings: AppSettings, session_id: str | None = None) -> SessionWorkspace:
        safe_id = session_id or uuid4().hex
        if not safe_id.replace("-", "").isalnum():
            raise ValueError("session_id may contain only letters, numbers, and hyphens")
        root = settings.runtime_root / "sessions" / safe_id
        output_dir = root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        return cls(session_id=safe_id, root=root, output_dir=output_dir)

