$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ServiceRoot = Join-Path $ProjectRoot "graphrag_service"

Push-Location $ProjectRoot
try {
    python -m uv sync --frozen
}
finally {
    Pop-Location
}

Push-Location $ServiceRoot
try {
    python -m uv sync --frozen
}
finally {
    Pop-Location
}

Write-Host "Both locked environments are ready."
