# Gemma 4 12B IT QAT + MTP draft head for local speech-to-speech testing.
# Mirrors C:\llama.cpp\launch_gemma-4-12B-it-qat-MTP.ps1, except ctx is 16k.
# Set GEMMA_API_KEY to the llama-server API key before running.

param(
    [string]$LlamaCppDir = "C:\llama.cpp",
    [int]$CtxSize = 16384,
    [int]$Port = 8818,
    [string]$ApiKey = $env:GEMMA_API_KEY
)

$ErrorActionPreference = 'Stop'
if (-not $ApiKey) {
    Write-Error "Set GEMMA_API_KEY or pass -ApiKey before launching Gemma."
    exit 1
}

$exe = Join-Path $LlamaCppDir 'llama-server.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    Write-Error "llama-server.exe not found at: $exe"
    exit 1
}

Write-Host "Launching Gemma 4 12B QAT + MTP for speech-to-speech testing..." -ForegroundColor Cyan
Write-Host "  CTX:    $CtxSize | Batch: 512 | Cache: f16" -ForegroundColor DarkGray
Write-Host "  Port:   127.0.0.1:$Port" -ForegroundColor DarkGray

$arguments = @(
    '-m', 'D:\LMStudio\Models\unsloth\gemma-4-12B-it-qat-GGUF\gemma-4-12B-it-qat-UD-Q4_K_XL.gguf'
    '-md', 'D:\LMStudio\Models\Janvitos\gemma-4-12B-it-qat-assistant-MTP-Q8_0-GGUF\gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf'
    '--mmproj', 'D:\LMStudio\Models\unsloth\gemma-4-12B-it-qat-GGUF\mmproj-F32.gguf'
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
