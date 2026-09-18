# eNRRCrew — uv upgrade

This directory contains the upgraded, session-isolated Streamlit application. Its model,
dataset, prompt, and GraphRAG assets are included in the parent repository directory.

## Prerequisites

- Windows with Python managed by `uv`
- Docker Desktop using Linux containers for CSV analysis
- An OpenAI-compatible API key and base URL

## Install

```powershell
cd uv
python -m uv sync --frozen
python -m uv run --frozen python -c "import sklearn, matminer; print(sklearn.__version__)"
Copy-Item .env.example .env
```

The import preflight should print `1.5.1`. If it reports that `sklearn.__version__` cannot be
imported, repair the locked wheel and repeat the preflight:

```powershell
python -m uv sync --frozen --refresh-package scikit-learn --reinstall-package scikit-learn
python -m uv run --frozen python -c "import sklearn, matminer; print(sklearn.__version__)"
```

Edit `.env` or define the equivalent environment variables. The sidebar can temporarily
override the API key and base URL for one browser session; credentials are never written by
the application.

## Build the CSV sandbox

Start Docker Desktop, then run:

```powershell
python -m uv export --frozen --only-group sandbox --no-emit-project `
  --output-file docker/csv-sandbox/requirements.lock.txt
docker build -t enrrcrew-csv-sandbox:local .\docker\csv-sandbox
```

The sandbox image installs its hash-checked requirements exported from `uv.lock`. It has no
network, runs as a non-root user, receives input files as read-only mounts, and writes first to
a size-limited in-container tmpfs. Validated artifacts are then copied into the current session's
output directory. If Docker is unavailable, CSV code execution fails closed and never runs on
the host.

## Run

```powershell
python -m uv run --frozen streamlit run src/enrrcrew/app.py
```

The application exposes five workspaces:

1. GraphRAG dialogue and retrieval
2. NH3 yield prediction
3. Faradaic-efficiency prediction
4. Docker-isolated CSV analysis
5. Model-backed catalyst recommendation

## Catalyst recommendation

The recommendation workspace screens known systems from the curated CSV and can generate
deterministic, one-step hypotheses from strong historical seeds. A formal recommendation must
be classified as `High` by both persisted models and must not fall outside the empirical
applicability domain. Yield `predict_proba` values are shown as uncalibrated model scores; the
FE value is a signed nearest-centroid margin, not a probability.

Generated names describe element sets and morphology hypotheses, not exact stoichiometric
formulas. Novelty is measured only against the local CSV. GraphRAG evidence is retrieved on
demand for a selected candidate and does not affect numeric ranking. Absence from the local
dataset or graph is not evidence of global literature or patent novelty.

Each run is auditable under
`runtime/sessions/<session-id>/recommendations/<run-id>/`. These files contain the request,
ranked result and CSV export, but never API credentials. Recommendation remains available
without an API key; only the optional GraphRAG evidence button is disabled.

Open-ended chat questions are coordinated by `AgentManager`. Its routing and GraphRAG tool trace
is available in a collapsed **Agent trace** section without exposing credentials or host paths.

Prediction forms and natural-language extraction share the same validated `PredictionInput`
schema. Extracted values are always shown for review before model execution.

### Legacy GraphRAG compatibility

GraphRAG 0.3.6 declares LanceDB `<0.14`, but the 0.13 release artifacts are no longer
available from PyPI. The lock therefore contains an explicit `lancedb==0.14.0` override—the
smallest available release—while retaining the GraphRAG 0.3.6 query API. Treat the local
GraphRAG integration smoke test as a required release check.

## Verify

```powershell
python -m uv run ruff check .
python -m uv run pytest --cov=enrrcrew
```

The coverage configuration enforces a 90% minimum and emphasizes prediction preprocessing,
recommendation qualification, sandbox boundaries, and per-session filesystem isolation.

`ENRRCREW_ASSET_ROOT` defaults to the parent repository and should contain `models/`, `input/`,
`output/`, and `settings.yaml`. Runtime files are written below `runtime/sessions/<session-id>`;
the application performs no recursive cleanup or bulk deletion.

The persisted estimators identify scikit-learn 1.5.1 as their training/runtime version, so the
uv lock intentionally uses 1.5.1 even though the legacy requirements file listed 1.5.2.
