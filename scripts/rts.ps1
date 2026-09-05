$ErrorActionPreference = "Stop"

$LocalTarget = Join-Path $PSScriptRoot "local_realtime.ps1"
$TargetFile = Join-Path $PSScriptRoot "rts.target"

if (Test-Path -LiteralPath $LocalTarget) {
    $Target = $LocalTarget
}
elseif (Test-Path -LiteralPath $TargetFile) {
    $Target = (Get-Content -LiteralPath $TargetFile -Raw).Trim()
}
else {
    throw "Unable to find local_realtime.ps1 or installed target file: $TargetFile"
}

if (-not (Test-Path -LiteralPath $Target)) {
    throw "Configured realtime launcher does not exist: $Target"
}

& $Target @args
