from __future__ import annotations

import os

from .config import ServiceSettings
from .corpus import read_jsonl
from .runner import GraphRunner
from .store import ControlStore


def main() -> None:
    settings = ServiceSettings.from_environment()
    documents = read_jsonl(settings.corpus_path)
    if not documents:
        raise SystemExit("No canonical corpus found. Run scripts/migrate_baseline.py first.")
    api_key = os.getenv("GRAPHRAG_API_KEY", "")
    base_url = os.getenv("GRAPHRAG_BASE_URL", "https://api.openai.com/v1")
    store = ControlStore(settings.database_path, settings.knowledge_root)
    runner = GraphRunner(settings.knowledge_root, store)
    with store.writer():
        version = runner.build_release(documents, api_key, base_url, update=False)
    print(f"Published baseline {version} with {len(documents)} documents")


if __name__ == "__main__":
    main()
