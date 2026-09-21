from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICE_SOURCE = PROJECT_ROOT / "graphrag_service" / "src"
sys.path.insert(0, str(SERVICE_SOURCE))

from enrrcrew_rag.corpus import migrate_baseline  # noqa: E402


def main() -> None:
    source = PROJECT_ROOT / "knowledge" / "source"
    audit = migrate_baseline(
        source / "raw_corpus.xls",
        source / "merged_abstracts.txt",
        PROJECT_ROOT / "knowledge" / "corpus" / "documents.jsonl",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
