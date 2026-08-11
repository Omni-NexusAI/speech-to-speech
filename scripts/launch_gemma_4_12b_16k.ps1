# Gemma 4 12B IT QAT + MTP draft head for local speech-to-speech testing.
# All executable and model locations are explicit parameters or environment
# variables so the launcher remains portable across Windows workstations.

param(
    [string]$LlamaCppDir = $env:LLAMA_CPP_DIR,
    [string]$ModelPath = $env:GEMMA_MODEL_PATH,
    [string]$DraftModelPath = $env:GEMMA_DRAFT_MODEL_PATH,
    [string]$MmprojPath = $env:GEMMA_MMPROJ_PATH,
    [int]$CtxSize = 16384,
    [int]$Port = 8818,
    [string]$ApiKey = $env:GEMMA_API_KEY
)

$ErrorActionPreference = 'Stop'
if (-not $LlamaCppDir) {
    Write-Error "Set LLAMA_CPP_DIR or pass -LlamaCppDir."
    exit 1
}
if (-not $ModelPath) {
    Write-Error "Set GEMMA_MODEL_PATH or pass -ModelPath."
    exit 1
}
if (-not $DraftModelPath) {
    Write-Error "Set GEMMA_DRAFT_MODEL_PATH or pass -DraftModelPath."
    exit 1
}
if (-not $MmprojPath) {
    Write-Error "Set GEMMA_MMPROJ_PATH or pass -MmprojPath."
    exit 1
}
if (-not $ApiKey) {
    Write-Error "Set GEMMA_API_KEY or pass -ApiKey before launching Gemma."
    exit 1
}

$exe = Join-Path $LlamaCppDir 'llama-server.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    Write-Error "llama-server.exe not found at: $exe"
    exit 1
}
foreach ($asset in @($ModelPath, $DraftModelPath, $MmprojPath)) {
    if (-not (Test-Path -LiteralPath $asset -PathType Leaf)) {
        Write-Error "Required Gemma asset not found at: $asset"
        exit 1
    }
}

Write-Host "Launching Gemma 4 12B QAT + MTP for speech-to-speech testing..." -ForegroundColor Cyan
Write-Host "  CTX:    $CtxSize | Batch: 512 | Cache: f16" -ForegroundColor DarkGray
Write-Host "  Fit margin: 2560 MiB reserved for the isolated audio.cpp candidate" -ForegroundColor DarkGray
Write-Host "  Port:   127.0.0.1:$Port" -ForegroundColor DarkGray

$arguments = @(
    '-m', $ModelPath
    '-md', $DraftModelPath
    '--mmproj', $MmprojPath
    '--mmproj-offload'
    '--spec-type', 'draft-mtp'
    '--spec-draft-n-max', '3'
    '--spec-draft-p-min', '0.75'
    '--spec-draft-ngl', '99'
    '--temp', '0.3'
    '--top-k', '20'
    '--top-p', '0.8'
    '--min-p', '0.05'
    '--repeat-penalty', '1.0'
    '--host', '127.0.0.1'
    '--port', [string]$Port
    '--api-key', $ApiKey
    '--ctx-size', [string]$CtxSize
    '--fit', 'on'
    '--fit-ctx', '8192'
    '--fit-margin', '2560'
    '--batch-size', '512'
    '--ubatch-size', '512'
    '-ngl', '48'
    '--threads', '8'
    '--threads-batch', '24'
    '--cache-type-k', 'f16'
    '--cache-type-v', 'f16'
    '--flash-attn', 'on'
    '--parallel', '1'
    '-np', '1'
    '--metrics'
    '--slots'
)

& $exe @arguments
exit $LASTEXITCODE
