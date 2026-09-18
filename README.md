# eNRRCrew

eNRRCrew is a reproducible decision-support application for electrocatalytic nitrogen
reduction research. It combines local GraphRAG retrieval, persisted machine-learning models,
session-isolated Streamlit workflows, Docker-sandboxed CSV analysis, and constrained catalyst
recommendation.

The maintained application is in [`uv/`](uv/). This repository contains only the current
Streamlit implementation; the previous `appUI.py` and Chainlit application are intentionally
excluded.

## Publication

The work associated with eNRRCrew was published as:

Xu Hu, Suya Chen, Letian Chen, Huijuan Wang, Xu Zhang, and Zhen Zhou,
“Automating structure-activity analysis for electrochemical nitrogen reduction catalyst design
through multi-agent collaborations,” *National Science Review*, **12**(11), nwaf372 (2025).
[https://doi.org/10.1093/nsr/nwaf372](https://doi.org/10.1093/nsr/nwaf372)

## Current capabilities

1. GraphRAG dialogue and literature retrieval
2. NH3-yield classification
3. Faradaic-efficiency classification
4. Docker-isolated CSV analysis
5. Catalyst recommendation using known systems and constrained generated hypotheses

Recommendation results are experiment candidates for validation, not confirmed discoveries.
Yield model scores are uncalibrated classifier scores, FE scores are centroid margins, and
novelty is assessed only against the included local dataset and knowledge graph.

## Repository layout

```text
eNRRCrew/
├─ input/                 Curated eNRR dataset
├─ models/                Persisted FE and yield model assets
├─ output/                GraphRAG index used by the local query service
├─ prompts/               GraphRAG prompts
├─ settings.yaml          GraphRAG configuration
└─ uv/                    Maintained Python 3.12 application
```

## Install

Install a current [uv](https://docs.astral.sh/uv/) release and Docker Desktop, then run:

```powershell
git clone https://github.com/nkuhuxu/eNRRCrew.git
cd eNRRCrew\uv
uv python install 3.12.7
uv sync --frozen
uv run --frozen python -c "import sklearn, matminer; print(sklearn.__version__)"
Copy-Item .env.example .env
```

The import preflight should print `1.5.1`. If an interrupted or incomplete Windows installation
reports `cannot import name '__version__' from 'sklearn'`, repair only that locked package and run
the preflight again:

```powershell
python -m uv sync --frozen --refresh-package scikit-learn --reinstall-package scikit-learn
python -m uv run --frozen python -c "import sklearn, matminer; print(sklearn.__version__)"
```

API credentials are optional for prediction and recommendation. They are required only for LLM
and GraphRAG operations. Configure them in `uv/.env` or enter a temporary value in the Streamlit
sidebar:

```dotenv
GRAPHRAG_API_KEY=
GRAPHRAG_BASE_URL=https://api.openai.com/v1
```

Never commit the populated `.env` file. Session credentials are not written by the application.

## Build the CSV sandbox

```powershell
cd uv
uv export --frozen --only-group sandbox --no-emit-project `
  --output-file docker/csv-sandbox/requirements.lock.txt
docker build -t enrrcrew-csv-sandbox:local .\docker\csv-sandbox
```

Generated analysis code runs only in this non-root, network-disabled container. The application
does not fall back to host execution when Docker is unavailable.

## Run

```powershell
cd uv
uv run --frozen streamlit run src/enrrcrew/app.py
```

All user-generated files are written below `uv/runtime/sessions/<session-id>/`. Model, input,
and GraphRAG assets are read from the repository root by default.

## Verify

```powershell
cd uv
uv sync --frozen
uv run --frozen ruff check .
uv run --frozen pytest --cov=enrrcrew
```

The test suite includes predictor preprocessing, session isolation, Docker sandbox boundaries,
GraphRAG mocks, recommendation generation/ranking, persisted-model integration, and Streamlit
interaction tests. Coverage is enforced at 90%; the test command fails if the total drops below
that threshold.

The repository is self-contained after cloning: application code, model assets, the curated
dataset, GraphRAG prompts/configuration, and the current GraphRAG index are all included. Package
installation still requires access to the dependency sources referenced by `uv.lock`, and CSV
code execution requires Docker Desktop plus the sandbox image described above.

## Security and data notes

- `.env`, virtual environments, caches, runtime sessions, and generated code are ignored.
- The included GraphRAG index is read-only at application runtime.
- Recommendation novelty is local-only and must not be interpreted as a patent or global
  literature novelty opinion.
- Generated catalyst labels represent element-set and morphology hypotheses, not exact
  stoichiometric formulas.

## License

This project is distributed under the [MIT License](LICENSE).
