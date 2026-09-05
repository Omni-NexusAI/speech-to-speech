param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "speech-to-speech\bin"),
    [string]$TargetScript = (Join-Path $PSScriptRoot "local_realtime.ps1"),
    [ValidateSet("User", "Process", "None")][string]$PathScope = "User",
    [string]$ProfilePath = $PROFILE.CurrentUserAllHosts,
    [switch]$SkipProfileFunction,
    [switch]$InternalNoRun
)

$ErrorActionPreference = "Stop"
$ProfileBeginMarker = "# BEGIN speech-to-speech rts"
$ProfileEndMarker = "# END speech-to-speech rts"

function Normalize-PathEntry([string]$Path) {
    return ([System.IO.Path]::GetFullPath($Path)).TrimEnd([char[]]@("\", "/")).ToLowerInvariant()
}

function Get-PathEntries([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { return @() }
    return @($Value -split ";" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function Get-PathValue([string]$Scope) {
    if ($Scope -eq "Process") { return [string]$env:Path }
    return [string][Environment]::GetEnvironmentVariable("Path", "User")
}

function Set-PathValue([string]$Scope, [string]$Value) {
    if ($Scope -eq "Process") {
        $env:Path = $Value
        return
    }
    [Environment]::SetEnvironmentVariable("Path", $Value, "User")
}

function Add-PathEntry([string]$Scope, [string]$Entry) {
    if ($Scope -eq "None") { return $false }

    $current = Get-PathValue $Scope
    $entries = @(Get-PathEntries $current)
    $normalizedEntry = Normalize-PathEntry $Entry
    foreach ($item in $entries) {
        try {
            if ((Normalize-PathEntry $item) -eq $normalizedEntry) {
                if ($Scope -eq "User") { $null = Add-PathEntry "Process" $Entry }
                return $false
            }
        }
        catch { }
    }

    $updatedEntries = @($entries + $Entry)
    Set-PathValue $Scope ($updatedEntries -join ";")
    if ($Scope -eq "User") { $null = Add-PathEntry "Process" $Entry }
    return $true
}

function Read-ProfileText([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return [pscustomobject]@{
            Text = ""
            Encoding = [System.Text.UTF8Encoding]::new($true, $true)
        }
    }

    $utf8 = [System.Text.UTF8Encoding]::new($false, $true)
    try {
        $reader = [System.IO.StreamReader]::new($Path, $utf8, $true)
        try {
            $text = $reader.ReadToEnd()
            $encoding = $reader.CurrentEncoding
        }
        finally { $reader.Dispose() }
    }
    catch [System.Text.DecoderFallbackException] {
        $reader = [System.IO.StreamReader]::new($Path, [System.Text.Encoding]::Default, $false)
        try {
            $text = $reader.ReadToEnd()
            $encoding = $reader.CurrentEncoding
        }
        finally { $reader.Dispose() }
    }
    return [pscustomobject]@{ Text = $text; Encoding = $encoding }
}

function Write-ProfileText([string]$Path, [string]$Text, [System.Text.Encoding]$Encoding) {
    [System.IO.File]::WriteAllText($Path, $Text, $Encoding)
}

function Remove-ProfileBlock([string]$Text) {
    $pattern = "(?ms)^" + [regex]::Escape($ProfileBeginMarker) + "\r?\n.*?^" + [regex]::Escape($ProfileEndMarker) + "\r?\n?"
    return [regex]::Replace($Text, $pattern, "")
}

function Quote-PowerShellString([string]$Value) {
    return "'" + $Value.Replace("'", "''") + "'"
}

function Install-ProfileFunction([string]$Path, [string]$CommandPath) {
    if ([string]::IsNullOrWhiteSpace($Path)) { throw "ProfilePath cannot be empty." }
    $profileDir = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($profileDir)) {
        New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
    }

    $profileState = Read-ProfileText $Path
    $current = [string]$profileState.Text
    $clean = Remove-ProfileBlock $current
    $newline = if ($clean.Contains("`r`n")) { "`r`n" } elseif ($clean.Contains("`n")) { "`n" } else { "`r`n" }
    $quotedCommandPath = Quote-PowerShellString $CommandPath
    $block = @($ProfileBeginMarker, "function rts {", "    & $quotedCommandPath @args", "}", $ProfileEndMarker) -join $newline

    if ($clean.Length -gt 0) { $updated = $block + $newline + $clean }
    else { $updated = $block + $newline }
    Write-ProfileText $Path $updated $profileState.Encoding
}

if ($InternalNoRun) { return }

$ResolvedInstallDir = [System.IO.Path]::GetFullPath($InstallDir)
$ResolvedTarget = [System.IO.Path]::GetFullPath($TargetScript)

if (-not (Test-Path -LiteralPath $ResolvedTarget)) {
    throw "Realtime launcher not found: $ResolvedTarget"
}

New-Item -ItemType Directory -Force -Path $ResolvedInstallDir | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "rts.ps1") -Destination (Join-Path $ResolvedInstallDir "rts.ps1") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "rts.cmd") -Destination (Join-Path $ResolvedInstallDir "rts.cmd") -Force
Set-Content -LiteralPath (Join-Path $ResolvedInstallDir "rts.target") -Value $ResolvedTarget -Encoding UTF8
$InstalledCmd = Join-Path $ResolvedInstallDir "rts.cmd"

if (-not $SkipProfileFunction) {
    Install-ProfileFunction $ProfilePath $InstalledCmd
}

$pathChanged = Add-PathEntry $PathScope $ResolvedInstallDir
Write-Host "Installed rts to $ResolvedInstallDir"
Write-Host "Target launcher: $ResolvedTarget"
if ($SkipProfileFunction) {
    Write-Host "PowerShell profile function skipped."
}
else {
    Write-Host "PowerShell profile function installed to $ProfilePath"
}
if ($PathScope -eq "None") {
    Write-Host "PATH update skipped."
}
elseif ($pathChanged) {
    Write-Host "$PathScope PATH updated. Open a new terminal if this session does not see rts yet."
}
else {
    Write-Host "$PathScope PATH already contains $ResolvedInstallDir"
}
