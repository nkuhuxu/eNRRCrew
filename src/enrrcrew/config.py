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
    knowledge_root: Path
    rag_service_url: str
    rag_service_token: str
    admin_token: str
    api_key: str
    base_url: str
    chat_model: str
    manager_model: str
    agent_max_rounds: int
    agent_timeout: int
    sandbox_image: str
    sandbox_timeout: int

    @classmethod
    def from_environment(cls) -> AppSettings:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        asset_root = _resolve_path(
            os.getenv("ENRRCREW_ASSET_ROOT", "."), PROJECT_ROOT
        )
        runtime_root = _resolve_path(
            os.getenv("ENRRCREW_RUNTIME_ROOT", "./runtime"), PROJECT_ROOT
        )
        knowledge_root = _resolve_path(
            os.getenv("ENRRCREW_KNOWLEDGE_ROOT", "./knowledge"), PROJECT_ROOT
        )
        return cls(
            project_root=PROJECT_ROOT,
            asset_root=asset_root,
            runtime_root=runtime_root,
            knowledge_root=knowledge_root,
            rag_service_url=os.getenv(
                "ENRRCREW_RAG_SERVICE_URL", "http://127.0.0.1:8765"
            ).rstrip("/"),
            rag_service_token=os.getenv("ENRRCREW_RAG_SERVICE_TOKEN", ""),
            admin_token=os.getenv("ENRRCREW_ADMIN_TOKEN", ""),
            api_key=os.getenv("GRAPHRAG_API_KEY", ""),
            base_url=os.getenv("GRAPHRAG_BASE_URL", "https://api.openai.com/v1"),
            chat_model=os.getenv("ENRRCREW_CHAT_MODEL", "gpt-4o-mini"),
            manager_model=os.getenv("ENRRCREW_MANAGER_MODEL", "gpt-4o"),
            agent_max_rounds=max(
                2, int(os.getenv("ENRRCREW_AGENT_MAX_ROUNDS", "8"))
            ),
            agent_timeout=max(
                1, int(os.getenv("ENRRCREW_AGENT_TIMEOUT", "180"))
            ),
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

    def validate_assets(self) -> list[str]:
        required = [self.models_dir, self.input_dir, self.knowledge_root]
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
