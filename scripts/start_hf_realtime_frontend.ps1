param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 7862,
    [switch]$Open
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$UiRoot = Join-Path $RepoRoot "web\hf-realtime-voice"
$WorktreePython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$ConfigPath = Join-Path $RepoRoot "examples\local_gemma_fasterqwen3tts.json"

function Test-FrontendPython([string]$Candidate) {
    if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate)) { return $false }
    try {
        & $Candidate -c "import uvicorn" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch { return $false }
}

$SharedPython = $env:SPEECH_TO_SPEECH_PYTHON
if (-not $SharedPython) { $SharedPython = "C:\speech-to-speech\.venv\Scripts\python.exe" }
$Python = if (Test-FrontendPython $WorktreePython) { $WorktreePython } elseif (Test-FrontendPython $SharedPython) { $SharedPython } else { "python" }
$oldGemmaBaseUrl = $env:GEMMA_AUDIO_BASE_URL
$oldGemmaModel = $env:GEMMA_AUDIO_MODEL
$runtimeConfig = if (Test-Path -LiteralPath $ConfigPath) { Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json } else { $null }
if ($runtimeConfig) {
    $env:GEMMA_AUDIO_BASE_URL = [string]$runtimeConfig.gemma_audio_base_url
    $env:GEMMA_AUDIO_MODEL = [string]$runtimeConfig.gemma_audio_model_name
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
    $env:GEMMA_AUDIO_BASE_URL = $oldGemmaBaseUrl
    $env:GEMMA_AUDIO_MODEL = $oldGemmaModel
}
