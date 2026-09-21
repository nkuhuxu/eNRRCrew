from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SERVICE_ROOT = PACKAGE_ROOT.parents[1]
PROJECT_ROOT = SERVICE_ROOT.parent


def _path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    return Path(value).expanduser().resolve() if value else default.resolve()


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    project_root: Path
    knowledge_root: Path
    service_token: str
    admin_token: str
    host: str = "127.0.0.1"
    port: int = 8765

    @classmethod
    def from_environment(cls) -> ServiceSettings:
        return cls(
            project_root=PROJECT_ROOT,
            knowledge_root=_path("ENRRCREW_KNOWLEDGE_ROOT", PROJECT_ROOT / "knowledge"),
            service_token=os.getenv("ENRRCREW_RAG_SERVICE_TOKEN", ""),
            admin_token=os.getenv("ENRRCREW_ADMIN_TOKEN", ""),
            host="127.0.0.1",
            port=int(os.getenv("ENRRCREW_RAG_SERVICE_PORT", "8765")),
        )

    @property
    def corpus_path(self) -> Path:
        return self.knowledge_root / "corpus" / "documents.jsonl"

    @property
    def database_path(self) -> Path:
        return self.knowledge_root / "state" / "control.sqlite"

    @property
    def active_path(self) -> Path:
        return self.knowledge_root / "active.json"
