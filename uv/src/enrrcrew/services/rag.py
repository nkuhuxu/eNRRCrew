from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Literal

_ENV_LOCK = Lock()


@contextmanager
def _temporary_graphrag_environment(api_key: str, base_url: str) -> Iterator[None]:
    with _ENV_LOCK:
        previous_key = os.environ.get("GRAPHRAG_API_KEY")
        previous_url = os.environ.get("GRAPHRAG_BASE_URL")
        try:
            os.environ["GRAPHRAG_API_KEY"] = api_key
            os.environ["GRAPHRAG_BASE_URL"] = base_url
            yield
        finally:
            if previous_key is None:
                os.environ.pop("GRAPHRAG_API_KEY", None)
            else:
                os.environ["GRAPHRAG_API_KEY"] = previous_key
            if previous_url is None:
                os.environ.pop("GRAPHRAG_BASE_URL", None)
            else:
                os.environ["GRAPHRAG_BASE_URL"] = previous_url


class RagService:
    def __init__(self, asset_root: Path, config_path: Path, api_key: str, base_url: str):
        self.asset_root = asset_root.resolve()
        self.config_path = config_path.resolve()
        self.api_key = api_key
        self.base_url = base_url

    def search(
        self,
        question: str,
        mode: Literal["local", "global"] = "local",
        community: int = 0,
        response_type: str = "single paragraph",
    ) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if not self.api_key:
            raise ValueError("An API key is required for GraphRAG search")
        if not self.config_path.exists():
            raise FileNotFoundError(f"GraphRAG configuration not found: {self.config_path}")
        from graphrag.query.cli import run_global_search, run_local_search

        runner = run_local_search if mode == "local" else run_global_search
        with _temporary_graphrag_environment(self.api_key, self.base_url):
            result = runner(
                str(self.config_path),
                None,
                str(self.asset_root),
                community,
                response_type,
                True,
                question,
            )
        if isinstance(result, tuple):
            return str(result[0])
        return str(result)
