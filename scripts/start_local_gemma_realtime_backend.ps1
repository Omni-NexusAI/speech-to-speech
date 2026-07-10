param(
    [string]$ConfigPath = "examples\local_gemma_fasterqwen3tts.json",
    [string]$TtsContainer = "281c411e5cfe02d0b0d903f67cd3fb712ab38bc705eb3edbedd8a2041c78702b"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ConfigFullPath = Join-Path $RepoRoot $ConfigPath

if (-not (Test-Path $ConfigFullPath)) {
    throw "Config file not found: $ConfigFullPath"
}

Write-Host "Checking FasterQwen3TTS container..."
$containerState = docker inspect $TtsContainer --format "{{.State.Status}}" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Could not inspect FasterQwen3TTS container $TtsContainer"
}
if ($containerState -ne "running") {
    Write-Host "Starting FasterQwen3TTS container $TtsContainer"
    docker start $TtsContainer | Out-Host
}

Write-Host "Waiting for FasterQwen3TTS health on http://127.0.0.1:8881/health"
for ($i = 0; $i -lt 60; $i++) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8881/health" -TimeoutSec 5
        if ($health.status -eq "ok") {
            Write-Host "FasterQwen3TTS is healthy."
            break
        }
    }
    catch {
        Start-Sleep -Seconds 2
    }
}

Write-Host "Checking Gemma on http://127.0.0.1:8818/v1/models"
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:8818/v1/models" -TimeoutSec 10 | Out-Null
}
catch {
    throw "Gemma is not reachable at http://127.0.0.1:8818/v1. Start it before launching the realtime backend."
}

Push-Location $RepoRoot
try {
    if (-not $env:OPENAI_API_KEY) {
        $env:OPENAI_API_KEY = "local-llama-cpp"
    }
    Write-Host "Starting speech-to-speech realtime backend on ws://127.0.0.1:8765/v1/realtime"
    uv run python -m speech_to_speech.s2s_pipeline $ConfigFullPath
}
finally {
    Pop-Location
}
