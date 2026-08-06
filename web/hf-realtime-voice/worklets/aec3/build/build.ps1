param(
    [switch]$SkipSmoke
)

$ErrorActionPreference = 'Stop'
$buildRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$assetRoot = Split-Path -Parent $buildRoot
$target = 'wasm32-unknown-unknown'
$wasmName = 'hfrt_sonora_aec3_wasm.wasm'
$builtWasm = Join-Path $buildRoot "target\$target\release\$wasmName"
$assetWasm = Join-Path $assetRoot 'aec3.wasm'
$manifestPath = Join-Path $assetRoot 'aec3.manifest.json'

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw 'cargo is required to build the AEC3 WebAssembly artifact.'
}
if (-not ((rustup target list --installed) -contains $target)) {
    throw "Rust target $target is required. Install it explicitly with: rustup target add $target"
}

Push-Location $buildRoot
try {
    cargo build --locked --target $target --release
    if ($LASTEXITCODE -ne 0) { throw "AEC3 WASM build failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

Copy-Item -LiteralPath $builtWasm -Destination $assetWasm -Force
$sha256 = (Get-FileHash -LiteralPath $assetWasm -Algorithm SHA256).Hash.ToLowerInvariant()
$manifest = [ordered]@{
    available = $true
    abiVersion = 1
    engine = 'Sonora WebRTC M145 AEC3'
    sourceRevision = 'a024d6ef8351add55be5e8b1d6cc35f555787660'
    wasm = 'aec3.wasm'
    sha256 = $sha256
    frameMs = 10
    output = 'pcm_s16le/16000/mono'
    doubleTalkTelemetry = 'aec3-output-evidence'
}
$temporaryManifest = "$manifestPath.tmp"
$manifestJson = $manifest | ConvertTo-Json
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($temporaryManifest, $manifestJson + [Environment]::NewLine, $utf8NoBom)
Move-Item -LiteralPath $temporaryManifest -Destination $manifestPath -Force

if (-not $SkipSmoke) {
    node (Join-Path $buildRoot 'smoke.mjs') $assetWasm
    if ($LASTEXITCODE -ne 0) { throw "AEC3 WASM smoke failed with exit code $LASTEXITCODE" }
}

Write-Output "AEC3 WASM: $assetWasm"
Write-Output "SHA-256: $sha256"
