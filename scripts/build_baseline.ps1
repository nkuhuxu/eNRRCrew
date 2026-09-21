$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ServiceRoot = Join-Path $ProjectRoot "graphrag_service"

Push-Location $ProjectRoot
try {
    & (Join-Path $ServiceRoot ".venv\Scripts\python.exe") scripts\migrate_baseline.py
}
finally {
    Pop-Location
}

Push-Location $ServiceRoot
try {
    python -m uv run --frozen python -m enrrcrew_rag.baseline
}
finally {
    Pop-Location
}
