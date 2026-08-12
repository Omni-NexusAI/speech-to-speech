param(
    [ValidateSet("start", "stop", "restart", "status")][string]$Action = "status",
    [ValidateSet("all", "backend", "frontend")][string]$Component = "all",
    [switch]$Open,
    [switch]$InternalNoRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RuntimeRoot = Join-Path $RepoRoot ".runtime"
$LogRoot = Join-Path $RuntimeRoot "logs"
$StatePath = Join-Path $RuntimeRoot "state.json"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
# Worktrees intentionally do not duplicate the sizeable local virtual
# environment. The tracked launcher still starts this worktree's source,
# while using the established local runtime when its own .venv is absent.
if (-not (Test-Path $Python)) {
    $SharedPython = $env:SPEECH_TO_SPEECH_PYTHON
    if (-not $SharedPython) { $SharedPython = "C:\speech-to-speech\.venv\Scripts\python.exe" }
    if (Test-Path $SharedPython) { $Python = $SharedPython }
}
$ConfigPath = Join-Path $RepoRoot "examples\local_gemma_fasterqwen3tts.json"
$ExpectedTtsModel = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
$StateVersion = 3
$Specs = @{
    backend = @{
        Port = 8765
        WorkDir = $RepoRoot
        Args = @("-m", "speech_to_speech.s2s_pipeline", $ConfigPath)
        Health = "http://127.0.0.1:8765/v1/pool"
        CommandTokens = @("-m speech_to_speech.s2s_pipeline", $ConfigPath)
    }
    frontend = @{
        Port = 7862
        WorkDir = (Join-Path $RepoRoot "web\hf-realtime-voice")
        Args = @("-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "7862")
        Health = "http://127.0.0.1:7862/api/config"
        CommandTokens = @("-m uvicorn server:app", "--port 7862")
    }
}

function Get-UiAssetGeneration {
    $indexPath = Join-Path $RepoRoot "web\hf-realtime-voice\index.html"
    $mainPath = Join-Path $RepoRoot "web\hf-realtime-voice\main.js"
    $wsPath = Join-Path $RepoRoot "web\hf-realtime-voice\ws\s2s-ws-client.js"
    try {
        $index = Get-Content -LiteralPath $indexPath -Raw
        $main = Get-Content -LiteralPath $mainPath -Raw
        $ws = Get-Content -LiteralPath $wsPath -Raw
        $mainVersion = [regex]::Match($index, 'main\.js\?v=([A-Za-z0-9._-]+)').Groups[1].Value
        $wsVersion = [regex]::Match($main, 's2s-ws-client\.js\?v=([A-Za-z0-9._-]+)').Groups[1].Value
        $chatVersion = [regex]::Match($main, 'chat\.js\?v=([A-Za-z0-9._-]+)').Groups[1].Value
        $playbackVersion = [regex]::Match($ws, 'audio-playback\.js\?v=([A-Za-z0-9._-]+)').Groups[1].Value
        if (@($mainVersion, $wsVersion, $chatVersion, $playbackVersion) -contains "") { return "unknown" }
        return "main=$mainVersion;ws=$wsVersion;chat=$chatVersion;playback=$playbackVersion"
    }
    catch { return "unknown" }
}

function Test-RuntimeSourcePath([string]$RelativePath) {
    if ([string]::IsNullOrWhiteSpace($RelativePath)) { return $false }
    $normalized = $RelativePath.Replace("\", "/").TrimStart("./").ToLowerInvariant()
    if ($normalized -match '(^|/)(tests?|testdata|fixtures|build|dist|node_modules|__pycache__)/') { return $false }
    if ($normalized -match '(^|/)(agents|readme|context)(\.[^/]*)?$') { return $false }
    $extension = [IO.Path]::GetExtension($normalized)
    return @(".py", ".js", ".mjs", ".html", ".css", ".json", ".wasm", ".ps1") -contains $extension
}

function Get-SourceIdentity {
    $revision = "unknown"
    $dirty = $true
    $paths = @()
    $enumerationOk = $false
    try {
        $revisionOutput = @(& git -c "safe.directory=$RepoRoot" -C $RepoRoot rev-parse HEAD 2>$null)
        if ($LASTEXITCODE -eq 0 -and $revisionOutput.Count -gt 0) { $revision = ([string]$revisionOutput[0]).Trim().ToLowerInvariant() }
        $enumeratedPaths = @(
            & git -c "safe.directory=$RepoRoot" -C $RepoRoot ls-files --cached --others --exclude-standard -- src examples/local_gemma_fasterqwen3tts.json scripts/local_realtime.ps1 web/hf-realtime-voice 2>$null |
                Where-Object { Test-RuntimeSourcePath ([string]$_) } |
                Sort-Object -Unique
        )
        $enumerationOk = $LASTEXITCODE -eq 0
        if ($enumerationOk) { $paths = $enumeratedPaths }
        $status = @(
            & git -c "safe.directory=$RepoRoot" -C $RepoRoot status --porcelain=v1 --untracked-files=all -- src examples/local_gemma_fasterqwen3tts.json scripts/local_realtime.ps1 web/hf-realtime-voice 2>$null |
                Where-Object {
                    $statusPath = ([string]$_).Substring([math]::Min(3, ([string]$_).Length))
                    if ($statusPath.Contains(" -> ")) { $statusPath = ($statusPath -split " -> ")[-1] }
                    Test-RuntimeSourcePath $statusPath.Trim('"')
                }
        )
        if ($LASTEXITCODE -eq 0) { $dirty = $status.Count -gt 0 }
    }
    catch { }

    $sourceEntries = @()
    foreach ($relativePath in $paths) {
        $fullPath = Join-Path $RepoRoot ([string]$relativePath)
        if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) { continue }
        $fileHash = (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $sourceEntries += (([string]$relativePath).Replace("\", "/") + "=" + $fileHash)
    }
    $fingerprint = "unknown"
    if ($enumerationOk -and $paths.Count -gt 0 -and $sourceEntries.Count -gt 0) {
        $sha = [Security.Cryptography.SHA256]::Create()
        try {
            $bytes = [Text.Encoding]::UTF8.GetBytes(($sourceEntries -join "`n"))
            $fingerprint = -join ($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString("x2") })
        }
        finally { $sha.Dispose() }
    }
    return [pscustomobject]@{
        revision = $revision
        dirty = [bool]$dirty
        fingerprint = $fingerprint
        uiAssetGeneration = Get-UiAssetGeneration
    }
}

function Get-Endpoint-Runtime([string]$Name) {
    try {
        $payload = Invoke-RestMethod -Uri $Specs[$Name].Health -TimeoutSec 3
        return $payload.runtime
    }
    catch { return $null }
}

function Test-SourceIdentityMatch($Actual, $Expected) {
    if (-not $Actual -or -not $Expected) { return $false }
    $revisionPattern = '^[0-9a-fA-F]{7,64}$'
    $fingerprintPattern = '^[0-9a-fA-F]{64}$'
    $assetToken = '[A-Za-z0-9._-]+'
    $assetPattern = "^main=$assetToken;ws=$assetToken;chat=$assetToken;playback=$assetToken`$"
    $actualRevision = [string]$Actual.source_revision
    $expectedRevision = [string]$Expected.revision
    $actualFingerprint = [string]$Actual.source_fingerprint
    $expectedFingerprint = [string]$Expected.fingerprint
    $actualAssets = [string]$Actual.ui_asset_generation
    $expectedAssets = [string]$Expected.uiAssetGeneration
    if (
        $actualRevision -notmatch $revisionPattern -or $expectedRevision -notmatch $revisionPattern -or
        $actualFingerprint -notmatch $fingerprintPattern -or $expectedFingerprint -notmatch $fingerprintPattern -or
        $actualAssets -notmatch $assetPattern -or $expectedAssets -notmatch $assetPattern -or
        $Actual.source_dirty -isnot [bool] -or $Expected.dirty -isnot [bool]
    ) { return $false }
    return (
        $actualRevision -ceq $expectedRevision -and
        [bool]$Actual.source_dirty -eq [bool]$Expected.dirty -and
        $actualFingerprint -ceq $expectedFingerprint -and
        $actualAssets -ceq $expectedAssets
    )
}

function Assert-ValidSourceIdentity($Identity) {
    $revisionPattern = '^[0-9a-f]{7,64}$'
    $fingerprintPattern = '^[0-9a-f]{64}$'
    $assetToken = '[A-Za-z0-9._-]+'
    $assetPattern = "^main=$assetToken;ws=$assetToken;chat=$assetToken;playback=$assetToken`$"
    if (
        -not $Identity -or
        [string]$Identity.revision -cnotmatch $revisionPattern -or
        $Identity.dirty -isnot [bool] -or
        [string]$Identity.fingerprint -cnotmatch $fingerprintPattern -or
        [string]$Identity.uiAssetGeneration -cnotmatch $assetPattern -or
        ([string]$Identity.uiAssetGeneration).Length -gt 256
    ) { throw "Runtime source identity is incomplete; refusing managed start." }
}

function Format-SourceIdentity($Identity) {
    if (-not $Identity) { return "unavailable" }
    $fingerprint = [string]$(if ($Identity.fingerprint) { $Identity.fingerprint } else { $Identity.source_fingerprint })
    if (-not $fingerprint) { $fingerprint = "unknown" }
    if ($fingerprint.Length -gt 12) { $fingerprint = $fingerprint.Substring(0, 12) }
    $revision = [string]$(if ($Identity.revision) { $Identity.revision } else { $Identity.source_revision })
    if (-not $revision) { $revision = "unknown" }
    if ($revision.Length -gt 12) { $revision = $revision.Substring(0, 12) }
    $dirtyValue = if ($null -ne $Identity.dirty) { $Identity.dirty } else { $Identity.source_dirty }
    $assets = [string]$(if ($Identity.uiAssetGeneration) { $Identity.uiAssetGeneration } else { $Identity.ui_asset_generation })
    if (-not $assets) { $assets = "unknown" }
    return "revision=$revision dirty=$dirtyValue fingerprint=$fingerprint assets=$assets"
}

function Read-State {
    if (-not (Test-Path $StatePath)) { return [pscustomobject]@{ version = $StateVersion } }
    try { return Get-Content $StatePath -Raw | ConvertFrom-Json }
    catch { Write-Warning "Ignoring unreadable runtime state: $StatePath"; return [pscustomobject]@{ version = $StateVersion } }
}

function Write-State($State) {
    New-Item -ItemType Directory -Force $RuntimeRoot, $LogRoot | Out-Null
    $State | Add-Member -Force NoteProperty version $StateVersion
    $State | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $StatePath
}

function Selected-Components {
    if ($Component -eq "all") { return @("backend", "frontend") }
    return @($Component)
}

function Selected-Components-Reversed {
    $items = @(Selected-Components)
    [array]::Reverse($items)
    return $items
}

function Endpoint-Ok([string]$Url) {
    try { Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3 | Out-Null; return $true }
    catch { return $false }
}

function Get-Process-Info([int]$ProcessId) {
    try {
        $item = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction Stop
        if ($item) {
            $start = $item.CreationDate
            if ($start -isnot [datetime]) { $start = [Management.ManagementDateTimeConverter]::ToDateTime([string]$start) }
            return [pscustomobject]@{
                pid = [int]$item.ProcessId
                parentPid = [int]$item.ParentProcessId
                executable = [string]$item.ExecutablePath
                commandLine = [string]$item.CommandLine
                startTimeUtc = $start.ToUniversalTime().ToString("o")
            }
        }
    }
    catch { }

    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $process) { return $null }
    $fallbackStart = $null
    try {
        if ($process.StartTime) { $fallbackStart = $process.StartTime.ToUniversalTime().ToString("o") }
    }
    catch { }
    return [pscustomobject]@{
        pid = [int]$process.Id
        parentPid = 0
        executable = [string]$process.Path
        commandLine = ""
        startTimeUtc = $fallbackStart
    }
}

function Get-Listener-Info([string]$Name) {
    $port = [int]$Specs[$Name].Port
    $owners = @()
    try {
        $owners = @(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique)
    }
    catch { }
    if ($owners.Count -eq 0) {
        # Non-elevated Windows sessions can be denied Get-NetTCPConnection even
        # for this user's Python processes. netstat still exposes the listener PID.
        $portPattern = [regex]::Escape([string]$port)
        $owners = @(
            & netstat -ano -p tcp 2>$null |
                ForEach-Object {
                    if ($_ -match "^\s*TCP\s+\S+:$portPattern\s+\S+\s+LISTENING\s+(\d+)\s*$") {
                        [int]$Matches[1]
                    }
                } |
                Select-Object -Unique
        )
    }
    if ($owners.Count -gt 1) { throw "Port $($Specs[$Name].Port) has multiple listening owners: $($owners -join ', ')" }
    if ($owners.Count -eq 0) { return $null }
    return Get-Process-Info ([int]$owners[0])
}

function Test-Expected-Process([string]$Name, $Info) {
    if (-not $Info -or -not $Info.commandLine) { return $false }
    $command = $Info.commandLine.ToLowerInvariant()
    if (-not $command.Contains($RepoRoot.ToLowerInvariant())) { return $false }
    foreach ($token in $Specs[$Name].CommandTokens) {
        if (-not $command.Contains(([string]$token).ToLowerInvariant())) { return $false }
    }
    return $true
}

function Test-Same-ProcessInstance($Left, $Right) {
    if (-not $Left -or -not $Right -or [int]$Left.pid -ne [int]$Right.pid) { return $false }
    try {
        return [math]::Abs((([datetime]$Left.startTimeUtc).ToUniversalTime() - ([datetime]$Right.startTimeUtc).ToUniversalTime()).TotalSeconds) -lt 1
    }
    catch { return [string]$Left.startTimeUtc -eq [string]$Right.startTimeUtc }
}

function Test-Verified-Descendant($Candidate, $Ancestor) {
    if (-not $Candidate -or -not $Ancestor) { return $false }
    if (Test-Same-ProcessInstance $Candidate $Ancestor) { return $true }
    $current = $Candidate
    $seen = @{}
    for ($depth = 0; $depth -lt 16; $depth++) {
        $parentPid = [int]$current.parentPid
        if ($parentPid -le 0 -or $seen.ContainsKey($parentPid)) { return $false }
        $seen[$parentPid] = $true
        $parent = Get-Process-Info $parentPid
        if (-not $parent) { return $false }
        if (Test-Same-ProcessInstance $parent $Ancestor) { return $true }
        $current = $parent
    }
    return $false
}

function Test-Started-Listener-Ownership([string]$Name, $Listener, $Launcher) {
    if (-not $Listener -or -not $Launcher) { return $false }
    # After Start-Process, command-line similarity is not ownership proof: an
    # independent matching process can win the port race. Only the exact
    # launcher instance or a verified descendant belongs to this transaction.
    return Test-Verified-Descendant $Listener $Launcher
}

function Test-Recorded-Listener($Record, $Listener) {
    if (-not $Record -or -not $Listener) { return $false }
    $recordPid = if ($Record.listenerPid) { [int]$Record.listenerPid } elseif ($Record.pid) { [int]$Record.pid } else { 0 }
    $recordStart = if ($Record.listenerStartTimeUtc) { [string]$Record.listenerStartTimeUtc } else { [string]$Record.startTimeUtc }
    $sameExecutable = -not $Record.listenerExecutable -or $Record.listenerExecutable -eq $Listener.executable
    try {
        $sameStart = [math]::Abs((([datetime]$recordStart).ToUniversalTime() - ([datetime]$Listener.startTimeUtc).ToUniversalTime()).TotalSeconds) -lt 1
    }
    catch { $sameStart = $recordStart -eq $Listener.startTimeUtc }
    return $recordPid -eq $Listener.pid -and $sameStart -and $sameExecutable
}

function Set-Component-State([string]$Name, $State, $Listener, $Launcher, [string]$Ownership) {
    if (-not $Launcher) { $Launcher = $Listener }
    $record = [pscustomobject]@{
        component = $Name
        ownership = $Ownership
        port = $Specs[$Name].Port
        command = ($Specs[$Name].Args -join " ")
        listenerPid = $Listener.pid
        listenerParentPid = $Listener.parentPid
        listenerExecutable = $Listener.executable
        listenerCommandLine = $Listener.commandLine
        listenerStartTimeUtc = $Listener.startTimeUtc
        launcherPid = $Launcher.pid
        launcherParentPid = $Launcher.parentPid
        launcherExecutable = $Launcher.executable
        launcherCommandLine = $Launcher.commandLine
        launcherStartTimeUtc = $Launcher.startTimeUtc
        stdout = (Join-Path $LogRoot "$Name.stdout.log")
        stderr = (Join-Path $LogRoot "$Name.stderr.log")
        sourceRevision = $script:LaunchSourceIdentity.revision
        sourceDirty = $script:LaunchSourceIdentity.dirty
        sourceFingerprint = $script:LaunchSourceIdentity.fingerprint
        uiAssetGeneration = $script:LaunchSourceIdentity.uiAssetGeneration
    }
    $State | Add-Member -Force NoteProperty $Name $record
    Write-State $State
}

function Adopt-Component([string]$Name, $State, $Listener) {
    $parent = Get-Process-Info $Listener.parentPid
    $launcher = if (Test-Expected-Process $Name $parent) { $parent } else { $Listener }
    Set-Component-State $Name $State $Listener $launcher "adopted"
    Write-Host "$Name adopted as managed (listener PID $($Listener.pid))."
}

function Verify-Dependencies {
    if (-not (Test-Path $Python)) { throw "Repo Python not found: $Python" }
    if (-not (Test-Path $ConfigPath)) { throw "Config file not found: $ConfigPath" }
    $config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
    $gemmaBase = ([string]$config.gemma_audio_base_url).TrimEnd("/")
    if (-not (Endpoint-Ok "$gemmaBase/models")) {
        Write-Warning "Local Gemma is not reachable at $gemmaBase. The pipeline will start model-free and re-check the selected provider per session."
    }
    if (-not (Endpoint-Ok "http://127.0.0.1:8881/health")) {
        Write-Warning "FasterQwen3TTS is not reachable on port 8881. The pipeline will still start; choose or start a TTS provider later."
    }
}

function Start-One([string]$Name, $State) {
    $spec = $Specs[$Name]
    $listener = Get-Listener-Info $Name
    if ($listener) {
        if (Test-Recorded-Listener $State.$Name $listener) {
            if (-not (Endpoint-Ok $spec.Health)) { throw "$Name has a recorded listener on port $($spec.Port), but its health endpoint failed." }
            $actualIdentity = Get-Endpoint-Runtime $Name
            if (-not (Test-SourceIdentityMatch $actualIdentity $script:LaunchSourceIdentity)) {
                throw "$Name is healthy but serves stale source identity. Expected $(Format-SourceIdentity $script:LaunchSourceIdentity); running $(Format-SourceIdentity $actualIdentity)."
            }
            Write-Host "$Name already running (listener PID $($listener.pid))."
            return
        }
        if (-not (Test-Expected-Process $Name $listener)) {
            throw "Refusing to start ${Name}: port $($spec.Port) is owned by unknown PID $($listener.pid)."
        }
        if (-not (Endpoint-Ok $spec.Health)) { throw "$Name has a matching listener on port $($spec.Port), but its health endpoint failed." }
        $actualIdentity = Get-Endpoint-Runtime $Name
        if (-not (Test-SourceIdentityMatch $actualIdentity $script:LaunchSourceIdentity)) {
            throw "$Name matches the command but serves stale source identity. Refusing adoption."
        }
        Adopt-Component $Name $State $listener
        return
    }

    New-Item -ItemType Directory -Force $LogRoot | Out-Null
    $stdout = Join-Path $LogRoot "$Name.stdout.log"
    $stderr = Join-Path $LogRoot "$Name.stderr.log"
    Set-Content -Path $stdout -Value ""
    Set-Content -Path $stderr -Value ""
    $oldUnbuffered = $env:PYTHONUNBUFFERED
    $oldRuntimeLog = $env:S2S_RUNTIME_LOG_FILE
    $oldPythonPath = $env:PYTHONPATH
    $oldRuntimeRevision = $env:S2S_RUNTIME_REVISION
    $oldRuntimeDirty = $env:S2S_RUNTIME_DIRTY
    $oldRuntimeFingerprint = $env:S2S_RUNTIME_SOURCE_FINGERPRINT
    $oldUiAssetGeneration = $env:S2S_UI_ASSET_GENERATION
    $env:PYTHONUNBUFFERED = "1"
    # Discard any inherited checkout path. The backend still needs this
    # worktree's src directory when it borrows the shared virtual environment.
    $env:PYTHONPATH = if ($Name -eq "backend") { Join-Path $RepoRoot "src" } else { "" }
    if ($Name -eq "backend") { $env:S2S_RUNTIME_LOG_FILE = $stdout }
    if ($Name -eq "backend" -and -not $env:OPENAI_API_KEY) { $env:OPENAI_API_KEY = "local-llama-cpp" }
    $env:S2S_RUNTIME_REVISION = $script:LaunchSourceIdentity.revision
    $env:S2S_RUNTIME_DIRTY = $(if ($script:LaunchSourceIdentity.dirty) { "1" } else { "0" })
    $env:S2S_RUNTIME_SOURCE_FINGERPRINT = $script:LaunchSourceIdentity.fingerprint
    $env:S2S_UI_ASSET_GENERATION = $script:LaunchSourceIdentity.uiAssetGeneration
    try {
        # Some parent launchers expose both `Path` and `PATH` in the Windows
        # process environment. Start-Process builds a case-insensitive child
        # dictionary and aborts on that duplicate, so normalize only the key
        # casing while preserving the effective search path value.
        $processEnvironment = [Environment]::GetEnvironmentVariables("Process")
        $pathKeys = @($processEnvironment.Keys | Where-Object { [string]$_ -ieq "Path" })
        if ($pathKeys.Count -gt 1) {
            $pathValue = [Environment]::GetEnvironmentVariable("Path", "Process")
            foreach ($pathKey in $pathKeys) {
                [Environment]::SetEnvironmentVariable([string]$pathKey, $null, "Process")
            }
            [Environment]::SetEnvironmentVariable("Path", $pathValue, "Process")
        }
        $launcher = Start-Process -FilePath $Python -ArgumentList $spec.Args -WorkingDirectory $spec.WorkDir -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    }
    finally {
        $env:PYTHONUNBUFFERED = $oldUnbuffered
        $env:S2S_RUNTIME_LOG_FILE = $oldRuntimeLog
        $env:PYTHONPATH = $oldPythonPath
        $env:S2S_RUNTIME_REVISION = $oldRuntimeRevision
        $env:S2S_RUNTIME_DIRTY = $oldRuntimeDirty
        $env:S2S_RUNTIME_SOURCE_FINGERPRINT = $oldRuntimeFingerprint
        $env:S2S_UI_ASSET_GENERATION = $oldUiAssetGeneration
    }

    for ($i = 0; $i -lt 90; $i++) {
        if (Endpoint-Ok $spec.Health) {
            $listener = Get-Listener-Info $Name
            $launcherInfo = Get-Process-Info $launcher.Id
            if ($listener -and -not (Test-Started-Listener-Ownership $Name $listener $launcherInfo)) {
                if ($launcherInfo) { Stop-Known-Process ([int]$launcherInfo.pid) }
                throw "$Name readiness found an unknown listener on port $($spec.Port). Refusing ownership without stopping that listener."
            }
            if ($listener) {
                $actualIdentity = Get-Endpoint-Runtime $Name
                if (-not (Test-SourceIdentityMatch $actualIdentity $script:LaunchSourceIdentity)) {
                    Stop-Known-Process ([int]$listener.pid)
                    if ($launcherInfo -and [int]$launcherInfo.pid -ne [int]$listener.pid) {
                        Stop-Known-Process ([int]$launcherInfo.pid)
                    }
                    throw "$Name started but its source identity does not match this checkout. Expected $(Format-SourceIdentity $script:LaunchSourceIdentity); running $(Format-SourceIdentity $actualIdentity)."
                }
                Set-Component-State $Name $State $listener $launcherInfo "started"
                Write-Host "$name ready (listener PID $($listener.pid), port $($spec.Port))."
                return
            }
        }
        Start-Sleep -Seconds 1
    }
    Get-Content $stderr -Tail 40 -ErrorAction SilentlyContinue | Out-Host
    throw "$Name failed readiness: $($spec.Health)"
}

function Stop-Known-Process([int]$ProcessId) {
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $process) { return }
    Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { return }
    try { Wait-Process -Id $ProcessId -Timeout 5 -ErrorAction Stop }
    catch {
        if (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
            Stop-Process -Id $ProcessId -Force
            Wait-Process -Id $ProcessId -Timeout 5 -ErrorAction SilentlyContinue
        }
    }
}

function Stop-One([string]$Name, $State) {
    $record = $State.$Name
    $listener = Get-Listener-Info $Name
    if ($listener -and -not (Test-Recorded-Listener $record $listener) -and -not (Test-Expected-Process $Name $listener)) {
        throw "Refusing to stop ${Name}: port $($Specs[$Name].Port) is owned by unknown PID $($listener.pid)."
    }
    if (-not $listener -and -not $record) { Write-Host "$Name is already stopped."; return }

    $launcher = $null
    if ($record -and $record.launcherPid) {
        $candidate = Get-Process-Info ([int]$record.launcherPid)
        if ($candidate -and (Test-Expected-Process $Name $candidate)) { $launcher = $candidate }
    }
    if (-not $launcher -and $listener) {
        $candidate = Get-Process-Info $listener.parentPid
        if ($candidate -and (Test-Expected-Process $Name $candidate)) { $launcher = $candidate }
    }

    if ($listener) { Stop-Known-Process $listener.pid }
    if ($launcher -and (!$listener -or $launcher.pid -ne $listener.pid)) { Stop-Known-Process $launcher.pid }
    $State.PSObject.Properties.Remove($Name)
    Write-State $State
    Write-Host "$Name stopped."
}

function Show-Status {
    $state = Read-State
    $currentIdentity = Get-SourceIdentity
    foreach ($name in @("backend", "frontend")) {
        $listener = Get-Listener-Info $name
        $record = $state.$name
        $health = Endpoint-Ok $Specs[$name].Health
        if (-not $listener) { $ownership = if ($record) { "stale-state" } else { "stopped" } }
        elseif (Test-Recorded-Listener $record $listener) { $ownership = "managed" }
        elseif (-not (Test-Expected-Process $name $listener)) { $ownership = "unknown-owner" }
        else { $ownership = "matching-legacy" }
        Write-Host ("{0}: {1}; listener={2}; endpoint={3}; port={4}" -f $name, $ownership, $(if($listener){"pid=$($listener.pid) started=$($listener.startTimeUtc)"}else{"none"}), $(if($health){"healthy"}else{"down"}), $Specs[$name].Port)
        if ($health) {
            $actualIdentity = Get-Endpoint-Runtime $name
            if (Test-SourceIdentityMatch $actualIdentity $currentIdentity) {
                Write-Host "$name source: current ($(Format-SourceIdentity $actualIdentity))"
            }
            else {
                Write-Warning "$name STALE SOURCE: checkout $(Format-SourceIdentity $currentIdentity); running $(Format-SourceIdentity $actualIdentity)"
            }
        }
    }
    $config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
    $gemmaBase = ([string]$config.gemma_audio_base_url).TrimEnd("/")
    Write-Host "Gemma: $(if(Endpoint-Ok "$gemmaBase/models"){'healthy'}else{'down'}) ($gemmaBase)"
    Write-Host "FasterQwen3TTS: $(if(Endpoint-Ok 'http://127.0.0.1:8881/health'){'healthy'}else{'down'}) ($ExpectedTtsModel)"
    Write-Host "Groxaxo: $(if(Endpoint-Ok 'http://127.0.0.1:8882/v1/backend/models'){'healthy'}else{'down'}) (user-managed)"
    Write-Host "Qwen3TTS audio.cpp: $(if(Endpoint-Ok 'http://127.0.0.1:8890/health'){'healthy'}else{'down'}) (isolated candidate)"
    if (Endpoint-Ok $Specs.backend.Health) {
        $pool = Invoke-RestMethod $Specs.backend.Health
        Write-Host "Backend runtime API: $($pool.runtime.api_version); started: $($pool.runtime.started_at_utc)"
    }
    if (Endpoint-Ok $Specs.frontend.Health) {
        $cfg = Invoke-RestMethod $Specs.frontend.Health
        Write-Host "UI API version: $($cfg.apiVersion)"
    }
    Write-Host "Context: 30 complete turns, compact_history=false; logs: $LogRoot"
}

$state = Read-State
if (-not $InternalNoRun) { switch ($Action) {
    "start" {
        $script:LaunchSourceIdentity = Get-SourceIdentity
        Assert-ValidSourceIdentity $script:LaunchSourceIdentity
        if ($Component -ne "frontend") { Verify-Dependencies }
        foreach ($name in Selected-Components) { Start-One $name $state }
        if ($Open) { Start-Process "http://127.0.0.1:7862" }
    }
    "stop" {
        foreach ($name in Selected-Components-Reversed) { Stop-One $name $state }
    }
    "restart" {
        $script:LaunchSourceIdentity = Get-SourceIdentity
        Assert-ValidSourceIdentity $script:LaunchSourceIdentity
        if ($Component -ne "frontend") { Verify-Dependencies }
        foreach ($name in Selected-Components-Reversed) { Stop-One $name $state }
        $state = Read-State
        foreach ($name in Selected-Components) { Start-One $name $state }
        if ($Open) { Start-Process "http://127.0.0.1:7862" }
    }
    "status" { Show-Status }
} }
