param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "speech-to-speech\bin"),
    [ValidateSet("User", "Process", "None")][string]$PathScope = "User",
    [switch]$RemovePath,
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

function Remove-PathEntry([string]$Scope, [string]$Entry) {
    if ($Scope -eq "None") { return $false }

    $current = Get-PathValue $Scope
    $entries = @(Get-PathEntries $current)
    $normalizedEntry = Normalize-PathEntry $Entry
    $kept = @()
    $removed = $false
    foreach ($item in $entries) {
        try {
            if ((Normalize-PathEntry $item) -eq $normalizedEntry) {
                $removed = $true
                continue
            }
        }
        catch { }
        $kept += $item
    }

    if ($removed) { Set-PathValue $Scope ($kept -join ";") }
    if ($Scope -eq "User") {
        $processRemoved = Remove-PathEntry "Process" $Entry
        return ($removed -or $processRemoved)
    }
    return $removed
}

function Read-ProfileText([string]$Path) {
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

function Uninstall-ProfileFunction([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path)) { return $false }
    $profileState = Read-ProfileText $Path
    $current = [string]$profileState.Text
    $updated = Remove-ProfileBlock $current
    if ($updated -eq $current) { return $false }
    Write-ProfileText $Path $updated $profileState.Encoding
    return $true
}

if ($InternalNoRun) { return }

$ResolvedInstallDir = [System.IO.Path]::GetFullPath($InstallDir)
foreach ($name in @("rts.ps1", "rts.cmd", "rts.target")) {
    Remove-Item -LiteralPath (Join-Path $ResolvedInstallDir $name) -Force -ErrorAction SilentlyContinue
}

Write-Host "Removed rts shims from $ResolvedInstallDir"
if (-not $SkipProfileFunction) {
    if (Uninstall-ProfileFunction $ProfilePath) {
        Write-Host "Removed PowerShell profile function from $ProfilePath"
    }
    else {
        Write-Host "PowerShell profile function not found in $ProfilePath"
    }
}
if ($RemovePath) {
    if (Remove-PathEntry $PathScope $ResolvedInstallDir) {
        Write-Host "Removed $ResolvedInstallDir from $PathScope PATH."
    }
    else {
        Write-Host "$PathScope PATH did not contain $ResolvedInstallDir."
    }
}
