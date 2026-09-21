from __future__ import annotations

import os
import time

from fastapi.testclient import TestClient

from enrrcrew_rag.app import create_app
from enrrcrew_rag.config import ServiceSettings

settings = ServiceSettings.from_environment()
required = {
    "GRAPHRAG_API_KEY": os.getenv("GRAPHRAG_API_KEY", ""),
    "ENRRCREW_RAG_SERVICE_TOKEN": settings.service_token,
    "ENRRCREW_ADMIN_TOKEN": settings.admin_token,
}
missing = [name for name, value in required.items() if not value]
if missing:
    raise SystemExit(f"Missing environment values: {', '.join(missing)}")

client = TestClient(create_app(settings))
headers = {
    "X-ENRRCREW-SERVICE-TOKEN": settings.service_token,
    "X-ENRRCREW-ADMIN-TOKEN": settings.admin_token,
    "X-ENRRCREW-UPSTREAM-KEY": required["GRAPHRAG_API_KEY"],
    "X-ENRRCREW-UPSTREAM-BASE-URL": os.getenv(
        "GRAPHRAG_BASE_URL", "https://api.openai.com/v1"
    ),
}
baseline = client.get("/health").json()["active_index"]
documents = [
    {
        "title": "Temporary incremental validation record A",
        "abstract": (
            "This synthetic validation abstract describes an Fe Mo nitrogen reduction catalyst "
            "under controlled neutral electrolyte conditions and exists only to verify incremental indexing. "
        ),
        "doi": "10.9999/enrrcrew.incremental.a",
        "publication_year": 2026,
        "source_row": 2,
    },
    {
        "title": "Temporary incremental validation record B",
        "abstract": (
            "This synthetic validation abstract describes a porous transition metal catalyst "
            "for ammonia synthesis and exists only to verify version activation and rollback behavior. "
        ),
        "doi": "10.9999/enrrcrew.incremental.b",
        "publication_year": 2026,
        "source_row": 3,
    },
]
batch_response = client.post(
    "/v1/batches",
    headers=headers,
    json={
        "session_id": "online-incremental-smoke",
        "source_file": "online_incremental_smoke.csv",
        "documents": documents,
    },
)
batch_response.raise_for_status()
batch = batch_response.json()
assert batch["accepted"] == 2

approval = client.post(
    f"/v1/batches/{batch['batch_id']}/approve", headers=headers
)
approval.raise_for_status()
job_id = approval.json()["job_id"]
for _ in range(2400):
    job = client.get(f"/v1/jobs/{job_id}", headers=headers).json()
    if job["status"] in {"published", "failed"}:
        break
    time.sleep(1)
else:
    raise TimeoutError("Incremental indexing did not finish in 40 minutes")
assert job["status"] == "published", job
assert job["version"] != baseline

rollback = client.post("/v1/versions/rollback", headers=headers)
rollback.raise_for_status()
assert rollback.json()["version"] == baseline
assert client.get("/health").json()["active_index"] == baseline
print(
    f"incremental_version={job['version']} rollback_version={baseline} "
    f"batch_id={batch['batch_id']}"
)
