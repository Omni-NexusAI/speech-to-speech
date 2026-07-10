param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 7862,
    [switch]$Open
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$UiRoot = Join-Path $RepoRoot "web\hf-realtime-voice"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

Push-Location $UiRoot
try {
    Write-Host "Starting HF Realtime Voice UI on http://${HostName}:${Port}"
    Write-Host "Speech-to-speech backend should be reachable from Settings as http://127.0.0.1:8765"
    if ($Open) {
        Start-Process "http://${HostName}:${Port}"
    }
    & $Python -m uvicorn server:app --host $HostName --port $Port
}
finally {
    Pop-Location
}
