# eNRRCrew Next 0.5.1

eNRRCrew Next is a self-contained Streamlit system for electrocatalytic nitrogen-reduction
literature retrieval, model-backed Yield/FE screening, catalyst recommendation, and isolated
CSV analysis. The GraphRAG runtime is separated from the persisted machine-learning runtime so
both dependency stacks remain reproducible.

The related work was published in *National Science Review* 12 (11), nwaf372:
[https://doi.org/10.1093/nsr/nwaf372](https://doi.org/10.1093/nsr/nwaf372).

## Architecture

```text
Streamlit / ML / AutoGen (Python 3.12, NumPy 1.26)
                 │ localhost HTTP + ephemeral token
                 ▼
GraphRAG 3.1.2 service (Python 3.12, NumPy 2 / Pandas 3 / PyArrow 25)
                 │
                 ▼
versioned corpus, immutable index releases, SQLite audit state
```

The repository contains two independent `pyproject.toml` and `uv.lock` files. The main
environment does not install GraphRAG or LanceDB. The GraphRAG service listens only on
`127.0.0.1`; `scripts/start.ps1` generates a temporary service token and passes it to both
processes without writing it to disk.

## Prerequisites

- Windows and Python 3.12.7 managed by `uv`
- Docker Desktop using Linux containers for CSV execution
- an OpenAI-compatible API key and base URL for LLM and GraphRAG operations
- an administrator token for knowledge-base publishing and rollback

Set credentials in the current PowerShell process or copy `.env.example` to `.env` for local
development. Do not commit `.env`.

```powershell
$env:GRAPHRAG_API_KEY = "your-key"
$env:GRAPHRAG_BASE_URL = "https://your-openai-compatible-endpoint/v1"
$env:ENRRCREW_ADMIN_TOKEN = "a-long-random-admin-secret"
```

The Streamlit sidebar can override the API key and base URL for the current browser session.
Credentials, service tokens, and administrator tokens are not stored in recommendation files,
corpus records, index audit downloads, or application logs.

## Installation and launch

From this directory:

```powershell
.\scripts\bootstrap.ps1
.\scripts\start.ps1
```

`bootstrap.ps1` synchronizes both locked environments. `start.ps1` starts the GraphRAG service
in a hidden window, waits for `/health`, then starts Streamlit. When Streamlit exits, the script
stops only the service process that it launched.

If Windows PowerShell blocks local scripts through its execution policy, use the supplied command
wrappers instead. They apply `ExecutionPolicy Bypass` only to the child PowerShell process and do
not change the machine or user policy:

```powershell
.\scripts\bootstrap.cmd
.\scripts\start.cmd
```

The application exposes six workspaces:

1. Dialogue and GraphRAG retrieval
2. NH3 Yield prediction
3. Faradaic-efficiency prediction
4. Docker-isolated CSV analysis
5. Deterministic catalyst recommendation
6. Knowledge-base upload, review, publishing, version activation, and rollback

The sidebar includes a compact guide to all six workspaces. The knowledge-base page also provides
a contextual runbook that recommends the next safe action and explains when each administrative
button should—and should not—be used.

GraphRAG unavailability disables retrieval and knowledge publishing only. Prediction,
recommendation, and CSV review remain available.

## First baseline index

The read-only source files are stored under `knowledge/source/`. The migration matches all 500
legacy abstracts to `raw_corpus.xls`, retains the longer record for the duplicate DOI
`10.1021/acsaem.3c01382`, and writes a deterministic 499-document JSONL corpus plus an audit
record. The remaining 1,819 unique workbook records are intentionally not imported.

After setting the API variables, build the baseline once:

```powershell
.\scripts\build_baseline.ps1
```

Each canonical document retains title, abstract, year, normalized DOI, content SHA-256, source
sheet/row, and revision. A document ID is based on normalized DOI or, when no DOI exists, the
content hash.

GraphRAG uses `gpt-4o-mini` for completion and `text-embedding-3-small` with an explicit vector
size of 1,536. Scientific abstracts are treated as untrusted data; the indexing and query
prompts reject instructions embedded in source text.

## Incremental publishing and rollback

The Knowledge base update page accepts CSV, XLS, and XLSX files up to 20 MB and 1,000 rows.
Title and abstract are required; DOI and publication year are optional. Common English and
Chinese column names are detected and can be corrected before submission.

The workflow is:

```text
uploaded → validated → needs_review → queued → preparing → indexing
                                                        → validating → publishing → published
                                                        └─────────────────────────→ failed
service restart while active                            └────────────→ interrupted
```

Duplicate DOI/content records are skipped. A matching DOI with different content is reported as
a conflict and never overwrites the existing document. Administrators approve batches using
`ENRRCREW_ADMIN_TOKEN`. Publishing runs under a single-writer lock and builds into
`knowledge/staging/`; the active release remains queryable until validation succeeds. Successful
builds are promoted into `knowledge/releases/<version>/` and `knowledge/active.json` is updated
atomically. A failed build leaves the current active version unchanged. Rollback changes only
the active pointer and does not delete an index.

Active jobs report stage-based progress, elapsed time, and a regularly updated heartbeat. The
page polls active jobs automatically while retaining manual refresh. Concurrent or repeated
approval is rejected, and unfinished jobs are marked `interrupted` after a service restart.

## Prediction, recommendation, and multi-Agent behavior

The persisted Yield and FE estimators share validated preprocessing and saved scalers. Catalyst
recommendations qualify only when both models return `High` and the candidate is not outside
the empirical applicability domain. Scores prioritize experiments; they are not calibrated
experimental success probabilities. Novelty is relative only to the local dataset and knowledge
graph, not the global literature or patent record.

Dialogue uses deterministic routing for single explicit tasks and a bounded AutoGen GroupChat
for open-ended or multi-part questions. `gpt-4o` coordinates and summarizes; `gpt-4o-mini`
specialists use narrowly scoped retrieval, Yield, FE, recommendation, or CSV-drafting tools.
Generated CSV code never executes from chat. It must be reviewed in the CSV page and can run
only inside the restricted Docker sandbox.

Build the sandbox image once:

```powershell
python -m uv export --frozen --only-group sandbox --no-emit-project `
  --output-file docker/csv-sandbox/requirements.lock.txt
docker build -t enrrcrew-csv-sandbox:local .\docker\csv-sandbox
```

## Verification

```powershell
python -m uv sync --frozen
python -m uv run --frozen ruff check .
python -m uv run --frozen pytest --cov=enrrcrew

Push-Location graphrag_service
python -m uv sync --frozen
python -m uv run --frozen ruff check .
python -m uv run --frozen pytest --cov=enrrcrew_rag
Pop-Location
```

Both projects enforce at least 90% test coverage. The service suite covers deterministic source
migration, validation and deduplication, authorization, single-writer publishing, atomic active
version switching, failure preservation, and rollback. The main suite retains prediction,
recommendation, multi-Agent, sandbox, session-isolation, and Streamlit regression tests.

The persisted estimators identify scikit-learn 1.5.1 as their compatible runtime version, so
the main lock intentionally uses 1.5.1.
