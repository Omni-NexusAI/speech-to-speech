param(
    [string]$ConfigPath = "examples\local_gemma_fasterqwen3tts.json"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ConfigFullPath = Join-Path $RepoRoot $ConfigPath
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $SharedPython = $env:SPEECH_TO_SPEECH_PYTHON
    if (-not $SharedPython) { $SharedPython = "C:\speech-to-speech\.venv\Scripts\python.exe" }
    if (Test-Path $SharedPython) { $Python = $SharedPython }
}

if (-not (Test-Path $ConfigFullPath)) {
    throw "Config file not found: $ConfigFullPath"
}
if (-not (Test-Path $Python)) {
    throw "Python environment not found. Create .venv or set SPEECH_TO_SPEECH_PYTHON."
}

$OldPythonPath = $env:PYTHONPATH
$OldOpenAiKey = $env:OPENAI_API_KEY
Push-Location $RepoRoot
try {
    $env:PYTHONPATH = Join-Path $RepoRoot "src"
    if (-not $env:OPENAI_API_KEY) {
        $env:OPENAI_API_KEY = "local-llama-cpp"
    }
    Write-Host "Starting speech-to-speech realtime backend on ws://127.0.0.1:8765/v1/realtime"
    Write-Host "Gemma and TTS services are user-managed and will be checked when a conversation starts."
    & $Python -m speech_to_speech.s2s_pipeline $ConfigFullPath
    if ($LASTEXITCODE -ne 0) {
        throw "Realtime backend exited with code $LASTEXITCODE"
    }
}
finally {
    $env:PYTHONPATH = $OldPythonPath
    $env:OPENAI_API_KEY = $OldOpenAiKey
    Pop-Location
}
