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
$FrontendUrl = "http://127.0.0.1:7862"
$WorktreePython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$SharedPython = $env:SPEECH_TO_SPEECH_PYTHON
if (-not $SharedPython) { $SharedPython = "C:\speech-to-speech\.venv\Scripts\python.exe" }

function Test-PythonRuntime([string]$Candidate, [string]$Name) {
    if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate)) { return $false }
    $oldPythonPath = $env:PYTHONPATH
    $oldErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 can promote a native process's harmless
        # stderr diagnostics to NativeCommandError while the script-wide
        # preference is Stop. Runtime readiness is determined by Python's exit
        # code, so keep warnings quiet without treating them as import failure.
        $ErrorActionPreference = "Continue"
        if ($Name -eq "backend") {
            $env:PYTHONPATH = Join-Path $RepoRoot "src"
            & $Candidate -c "import nltk, requests, torch, transformers, uvicorn; import speech_to_speech.s2s_pipeline" *> $null
        }
        else {
            $env:PYTHONPATH = ""
            & $Candidate -c "import fastapi, httpx, uvicorn" *> $null
        }
        return $LASTEXITCODE -eq 0
    }
    catch { return $false }
    finally {
        $env:PYTHONPATH = $oldPythonPath
        $ErrorActionPreference = $oldErrorActionPreference
    }
}

function Resolve-PythonRuntime([string]$Name) {
    foreach ($candidate in @($WorktreePython, $SharedPython)) {
        if (Test-PythonRuntime $candidate $Name) { return $candidate }
    }
    throw "No complete $Name Python runtime is available. Checked $WorktreePython and $SharedPython."
}

# Worktrees can contain a lightweight test-only .venv. Resolve each component
# only when it starts and validate that component's real import surface.
$ConfigPath = Join-Path $RepoRoot "examples\local_gemma_fasterqwen3tts.json"
$FrontendRoot = Join-Path $RepoRoot "web\hf-realtime-voice"
$UiSettingsPath = Join-Path $RuntimeRoot "hf_realtime_ui_settings.json"
$ExpectedTtsModel = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
$StateVersion = 2
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
        WorkDir = $FrontendRoot
        Args = @("-m", "uvicorn", "server:app", "--app-dir", $FrontendRoot, "--host", "127.0.0.1", "--port", "7862")
        Health = "http://127.0.0.1:7862/api/config"
        CommandTokens = @("-m uvicorn server:app", "--app-dir", $FrontendRoot, "--port 7862")
    }
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

function Resolve-GemmaRuntimeIdentity([string]$SettingsPath = $UiSettingsPath) {
    $config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
    $provider = "local"
    $baseUrl = [string]$config.gemma_audio_base_url
    $model = [string]$config.gemma_audio_model_name
    if (Test-Path -LiteralPath $SettingsPath) {
        try {
            $saved = Get-Content -LiteralPath $SettingsPath -Raw | ConvertFrom-Json
            $savedProvider = ([string]$saved.modelProvider).Trim().ToLowerInvariant()
            if ($savedProvider -in @("local", "remote")) { $provider = $savedProvider }
            if ($saved.modelName -and -not [string]::IsNullOrWhiteSpace([string]$saved.modelName)) {
                $model = ([string]$saved.modelName).Trim()
            }
            if ($provider -eq "remote" -and $saved.modelUrl) {
                $candidate = ([string]$saved.modelUrl).Trim()
                if ($candidate -match '^https?://') { $baseUrl = $candidate }
            }
        }
        catch { Write-Warning "Ignoring unreadable UI settings while resolving Gemma identity: $SettingsPath" }
    }
    return [pscustomobject]@{
        provider = $provider
        baseUrl = $baseUrl.TrimEnd("/")
        model = $model
    }
}

function Open-FrontendIfReady {
    if (-not (Endpoint-Ok $Specs.frontend.Health)) {
        throw "Cannot open HF Realtime Voice UI because the verified frontend is unavailable at $FrontendUrl."
    }
    Start-Process $FrontendUrl
}

function Write-FrontendUrlIfHealthy {
    # Starting or restarting just the backend must still leave a copyable UI
    # address when the separately managed frontend is already healthy.  This
    # is deliberately informational: Only -Open is allowed to launch a browser.
    if (Endpoint-Ok $Specs.frontend.Health) {
        Write-Host "HF Realtime Voice UI: $FrontendUrl"
        return $true
    }
    return $false
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

function Test-Recorded-Listener($Record, $Listener) {
    if (-not $Record -or -not $Listener) { return $false }
    $recordPid = if ($Record.listenerPid) { [int]$Record.listenerPid } elseif ($Record.pid) { [int]$Record.pid } else { 0 }
    $recordStart = if ($Record.listenerStartTimeUtc) { $Record.listenerStartTimeUtc } else { $Record.startTimeUtc }
    $sameExecutable = -not $Record.listenerExecutable -or $Record.listenerExecutable -eq $Listener.executable
    try {
        # ConvertFrom-Json can materialize an ISO timestamp as a local DateTime,
        # while the live CIM value remains a round-trip UTC string. Parse both
        # as DateTimeOffset so an identical instant is not shifted by the local
        # timezone and mislabeled as an unknown process owner.
        $style = [Globalization.DateTimeStyles]::RoundtripKind
        $culture = [Globalization.CultureInfo]::InvariantCulture
        $recordMoment = if ($recordStart -is [datetime]) {
            $recordStart.ToUniversalTime()
        }
        else {
            [datetimeoffset]::Parse([string]$recordStart, $culture, $style).UtcDateTime
        }
        $listenerStart = $Listener.startTimeUtc
        $listenerMoment = if ($listenerStart -is [datetime]) {
            $listenerStart.ToUniversalTime()
        }
        else {
            [datetimeoffset]::Parse([string]$listenerStart, $culture, $style).UtcDateTime
        }
        $sameStart = [math]::Abs(($recordMoment - $listenerMoment).TotalSeconds) -lt 1
    }
    catch { $sameStart = [string]$recordStart -eq [string]$Listener.startTimeUtc }
    return $recordPid -eq $Listener.pid -and $sameStart -and $sameExecutable
}

function Test-Launched-Listener([string]$Name, $Launcher, $Listener) {
    if (-not $Launcher -or -not $Listener) { return $false }
    if (Test-Recorded-Listener $Launcher $Listener) { return $true }
    if (-not (Test-Expected-Process $Name $Listener)) { return $false }
    if ([int]$Listener.parentPid -ne [int]$Launcher.pid) { return $false }
    try {
        $style = [Globalization.DateTimeStyles]::RoundtripKind
        $culture = [Globalization.CultureInfo]::InvariantCulture
        $launcherMoment = [datetimeoffset]::Parse([string]$Launcher.startTimeUtc, $culture, $style).UtcDateTime
        $listenerMoment = [datetimeoffset]::Parse([string]$Listener.startTimeUtc, $culture, $style).UtcDateTime
        $elapsedSeconds = ($listenerMoment - $launcherMoment).TotalSeconds
        return $elapsedSeconds -ge -1 -and $elapsedSeconds -le 15
    }
    catch { return $false }
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
    $null = Resolve-PythonRuntime "backend"
    if (-not (Test-Path $ConfigPath)) { throw "Config file not found: $ConfigPath" }
    $gemma = Resolve-GemmaRuntimeIdentity
    if (-not (Endpoint-Ok "$($gemma.baseUrl)/models")) {
        Write-Warning "Selected $($gemma.provider) Gemma is not reachable at $($gemma.baseUrl). The pipeline will start model-free and re-check the selected provider per session."
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
            Write-Host "$Name already running (listener PID $($listener.pid))."
            if ($Name -eq "frontend") { Write-FrontendUrlIfHealthy | Out-Null }
            return
        }
        if (-not (Test-Expected-Process $Name $listener)) {
            throw "Refusing to start ${Name}: port $($spec.Port) is owned by unknown PID $($listener.pid)."
        }
        if (-not (Endpoint-Ok $spec.Health)) { throw "$Name has a matching listener on port $($spec.Port), but its health endpoint failed." }
        Adopt-Component $Name $State $listener
        if ($Name -eq "frontend") { Write-FrontendUrlIfHealthy | Out-Null }
        return
    }

    $Python = Resolve-PythonRuntime $Name
    New-Item -ItemType Directory -Force $LogRoot | Out-Null
    $stdout = Join-Path $LogRoot "$Name.stdout.log"
    $stderr = Join-Path $LogRoot "$Name.stderr.log"
    Set-Content -Path $stdout -Value ""
    Set-Content -Path $stderr -Value ""
    $oldUnbuffered = $env:PYTHONUNBUFFERED
    $oldRuntimeLog = $env:S2S_RUNTIME_LOG_FILE
    $oldPythonPath = $env:PYTHONPATH
    $oldGemmaBaseUrl = $env:GEMMA_AUDIO_BASE_URL
    $oldGemmaModel = $env:GEMMA_AUDIO_MODEL
    $env:PYTHONUNBUFFERED = "1"
    # Discard any inherited checkout path. The backend still needs this
    # worktree's src directory when it borrows the shared virtual environment.
    $env:PYTHONPATH = if ($Name -eq "backend") { Join-Path $RepoRoot "src" } else { "" }
    if ($Name -eq "backend") { $env:S2S_RUNTIME_LOG_FILE = $stdout }
    if ($Name -eq "backend" -and -not $env:OPENAI_API_KEY) { $env:OPENAI_API_KEY = "local-llama-cpp" }
    if ($Name -eq "frontend") {
        $gemma = Resolve-GemmaRuntimeIdentity
        $env:GEMMA_AUDIO_BASE_URL = [string]$gemma.baseUrl
        $env:GEMMA_AUDIO_MODEL = [string]$gemma.model
    }
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
        $launcherStartTimeUtc = try { $launcher.StartTime.ToUniversalTime().ToString("o") } catch { [datetime]::UtcNow.ToString("o") }
        # A Windows venv launcher may create the base interpreter as its direct
        # child and exit before readiness. Record the exact process we created
        # immediately so that child ownership remains provable after that exit.
        $launcherInfo = [pscustomobject]@{
            pid = [int]$launcher.Id
            parentPid = 0
            executable = [string]$Python
            commandLine = ('"{0}" {1}' -f $Python, ($spec.Args -join " "))
            startTimeUtc = $launcherStartTimeUtc
        }
    }
    finally {
        $env:PYTHONUNBUFFERED = $oldUnbuffered
        $env:S2S_RUNTIME_LOG_FILE = $oldRuntimeLog
        $env:PYTHONPATH = $oldPythonPath
        $env:GEMMA_AUDIO_BASE_URL = $oldGemmaBaseUrl
        $env:GEMMA_AUDIO_MODEL = $oldGemmaModel
    }

    for ($i = 0; $i -lt 90; $i++) {
        if (Endpoint-Ok $spec.Health) {
            $listener = Get-Listener-Info $Name
            # Accept the exact Start-Process identity or its direct Windows
            # venv child. Expected command tokens, parent PID, and a bounded
            # start-time relationship reject an unrelated port-race winner.
            $isLaunchedListener = $listener -and (Test-Launched-Listener $Name $launcherInfo $listener)
            if ($isLaunchedListener) {
                Set-Component-State $Name $State $listener $launcherInfo "started"
                Write-Host "$name ready (listener PID $($listener.pid), port $($spec.Port))."
                if ($Name -eq "frontend") { Write-FrontendUrlIfHealthy | Out-Null }
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
    try {
        # Prefer the already-resolved process object.  On some Windows hosts,
        # Stop-Process -Id can throw an internal NullReferenceException when a
        # listener was launched from a different job/sandbox context.
        Stop-Process -InputObject $process -ErrorAction Stop
    }
    catch {
        # Process.Kill() is available in Windows PowerShell 5.1 and remains
        # scoped to the original ownership-validated Process object. Never
        # re-resolve by PID here: Windows could reuse an exited process's PID.
        try {
            if ($process.HasExited) { return }
            $process.Kill()
        }
        catch {
            try { if ($process.HasExited) { return } } catch { }
            throw "Failed to stop owned process PID ${ProcessId}: $($_.Exception.Message)"
        }
    }
    try {
        if ($process.HasExited) { return }
        if ($process.WaitForExit(5000)) { return }
    }
    catch {
        try { if ($process.HasExited) { return } } catch { }
    }
    try {
        if (-not $process.HasExited) {
            $process.Kill()
            [void]$process.WaitForExit(5000)
        }
    }
    catch {
        try { if ($process.HasExited) { return } } catch { }
        throw "Failed to force-stop owned process PID ${ProcessId}: $($_.Exception.Message)"
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
    foreach ($name in @("backend", "frontend")) {
        $listener = Get-Listener-Info $name
        $record = $state.$name
        $health = Endpoint-Ok $Specs[$name].Health
        if (-not $listener) { $ownership = if ($record) { "stale-state" } else { "stopped" } }
        elseif (Test-Recorded-Listener $record $listener) { $ownership = "managed" }
        elseif (-not (Test-Expected-Process $name $listener)) { $ownership = "unknown-owner" }
        else { $ownership = "matching-legacy" }
        Write-Host ("{0}: {1}; listener={2}; endpoint={3}; port={4}" -f $name, $ownership, $(if($listener){"pid=$($listener.pid) started=$($listener.startTimeUtc)"}else{"none"}), $(if($health){"healthy"}else{"down"}), $Specs[$name].Port)
        if ($name -eq "frontend" -and $health) { Write-FrontendUrlIfHealthy | Out-Null }
    }
    $gemma = Resolve-GemmaRuntimeIdentity
    Write-Host "Gemma: $(if(Endpoint-Ok "$($gemma.baseUrl)/models"){'healthy'}else{'down'}) ($($gemma.provider), $($gemma.model), $($gemma.baseUrl))"
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
    Write-Host "Context: conversation-scoped; see Realtime Diagnostics for the active compaction policy and budget; logs: $LogRoot"
}

$state = Read-State
if (-not $InternalNoRun) { switch ($Action) {
    "start" {
        if ($Component -ne "frontend") { Verify-Dependencies }
        foreach ($name in Selected-Components) { Start-One $name $state }
        if ($Component -eq "backend") { Write-FrontendUrlIfHealthy | Out-Null }
        if ($Open) { Open-FrontendIfReady }
    }
    "stop" {
        foreach ($name in Selected-Components-Reversed) { Stop-One $name $state }
    }
    "restart" {
        if ($Component -ne "frontend") { Verify-Dependencies }
        foreach ($name in Selected-Components-Reversed) { Stop-One $name $state }
        $state = Read-State
        foreach ($name in Selected-Components) { Start-One $name $state }
        if ($Component -eq "backend") { Write-FrontendUrlIfHealthy | Out-Null }
        if ($Open) { Open-FrontendIfReady }
    }
    "status" { Show-Status }
} }
