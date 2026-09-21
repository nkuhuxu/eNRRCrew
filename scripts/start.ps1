$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ServiceRoot = Join-Path $ProjectRoot "graphrag_service"
$ServicePython = Join-Path $ServiceRoot ".venv\Scripts\python.exe"

if (-not $env:ENRRCREW_RAG_SERVICE_TOKEN) {
    # Windows PowerShell 5.1 runs on .NET Framework, which does not provide
    # RandomNumberGenerator.GetBytes(int) or Convert.ToHexString().
    $TokenBytes = New-Object byte[] 32
    $RandomNumberGenerator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $RandomNumberGenerator.GetBytes($TokenBytes)
    }
    finally {
        $RandomNumberGenerator.Dispose()
    }
    $env:ENRRCREW_RAG_SERVICE_TOKEN = (
        [BitConverter]::ToString($TokenBytes).Replace("-", "").ToLowerInvariant()
    )
}
$env:ENRRCREW_KNOWLEDGE_ROOT = Join-Path $ProjectRoot "knowledge"
$env:ENRRCREW_ASSET_ROOT = $ProjectRoot
$env:ENRRCREW_RUNTIME_ROOT = Join-Path $ProjectRoot "runtime"

$Service = Start-Process -FilePath $ServicePython `
    -ArgumentList "-m", "uvicorn", "enrrcrew_rag.app:app", "--host", "127.0.0.1", "--port", "8765" `
    -WorkingDirectory $ServiceRoot -WindowStyle Hidden -PassThru

try {
    $Healthy = $false
    foreach ($Attempt in 1..60) {
        try {
            $Health = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -TimeoutSec 2
            if ($Health.status -eq "ok") {
                $Healthy = $true
                break
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $Healthy) {
        throw "GraphRAG service did not become healthy."
    }
    Push-Location $ProjectRoot
    try {
        python -m uv run --frozen streamlit run src/enrrcrew/app.py
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($Service -and -not $Service.HasExited) {
        Stop-Process -Id $Service.Id
    }
}
