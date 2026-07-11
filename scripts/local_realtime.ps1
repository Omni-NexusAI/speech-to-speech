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
$ConfigPath = Join-Path $RepoRoot "examples\local_gemma_fasterqwen3tts.json"
$TtsContainer = "281c411e5cfe02d0b0d903f67cd3fb712ab38bc705eb3edbedd8a2041c78702b"
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
        WorkDir = (Join-Path $RepoRoot "web\hf-realtime-voice")
        Args = @("-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "7862")
        Health = "http://127.0.0.1:7862/api/config"
        CommandTokens = @("-m uvicorn server:app", "--port 7862")
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

function Get-Process-Info([int]$ProcessId) {
    $item = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue
    if (-not $item) { return $null }
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

function Get-Listener-Info([string]$Name) {
    $owners = @(Get-NetTCPConnection -State Listen -LocalPort $Specs[$Name].Port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique)
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
    $recordStart = if ($Record.listenerStartTimeUtc) { [string]$Record.listenerStartTimeUtc } else { [string]$Record.startTimeUtc }
    return $recordPid -eq $Listener.pid -and $recordStart -eq $Listener.startTimeUtc
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
    if (-not (Test-Path $Python)) { throw "Repo Python not found: $Python" }
    if (-not (Test-Path $ConfigPath)) { throw "Config file not found: $ConfigPath" }
    $config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
    $gemmaBase = ([string]$config.gemma_audio_base_url).TrimEnd("/")
    if (-not (Endpoint-Ok "$gemmaBase/models")) { throw "Gemma is not reachable at $gemmaBase" }

    $containerState = docker inspect $TtsContainer --format "{{.State.Status}}" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "FasterQwen3TTS container not found: $TtsContainer" }
    if ($containerState -ne "running") { docker start $TtsContainer | Out-Null }
    $containerEnv = docker inspect $TtsContainer --format "{{json .Config.Env}}" | ConvertFrom-Json
    if ($containerEnv -notcontains "MODEL_ID=$ExpectedTtsModel") {
        throw "FasterQwen3TTS container is not configured for $ExpectedTtsModel"
    }
    if (-not (Endpoint-Ok "http://127.0.0.1:8881/health")) { throw "FasterQwen3TTS is not healthy on port 8881" }
}

function Start-One([string]$Name, $State) {
    $spec = $Specs[$Name]
    $listener = Get-Listener-Info $Name
    if ($listener) {
        if (-not (Test-Expected-Process $Name $listener)) {
            throw "Refusing to start ${Name}: port $($spec.Port) is owned by unknown PID $($listener.pid)."
        }
        if (-not (Endpoint-Ok $spec.Health)) { throw "$Name has a matching listener on port $($spec.Port), but its health endpoint failed." }
        if (Test-Recorded-Listener $State.$Name $listener) {
            Write-Host "$Name already running (listener PID $($listener.pid))."
            return
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
    $env:PYTHONUNBUFFERED = "1"
    if ($Name -eq "backend") { $env:S2S_RUNTIME_LOG_FILE = $stdout }
    if ($Name -eq "backend" -and -not $env:OPENAI_API_KEY) { $env:OPENAI_API_KEY = "local-llama-cpp" }
    try {
        $launcher = Start-Process -FilePath $Python -ArgumentList $spec.Args -WorkingDirectory $spec.WorkDir -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    }
    finally {
        $env:PYTHONUNBUFFERED = $oldUnbuffered
        $env:S2S_RUNTIME_LOG_FILE = $oldRuntimeLog
    }

    for ($i = 0; $i -lt 90; $i++) {
        if (Endpoint-Ok $spec.Health) {
            $listener = Get-Listener-Info $Name
            if ($listener -and (Test-Expected-Process $Name $listener)) {
                $launcherInfo = Get-Process-Info $launcher.Id
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
    if ($listener -and -not (Test-Expected-Process $Name $listener)) {
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
        elseif (-not (Test-Expected-Process $name $listener)) { $ownership = "unknown-owner" }
        elseif (Test-Recorded-Listener $record $listener) { $ownership = "managed" }
        else { $ownership = "matching-legacy" }
        Write-Host ("{0}: {1}; listener={2}; endpoint={3}; port={4}" -f $name, $ownership, $(if($listener){"pid=$($listener.pid) started=$($listener.startTimeUtc)"}else{"none"}), $(if($health){"healthy"}else{"down"}), $Specs[$name].Port)
    }
    $config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
    $gemmaBase = ([string]$config.gemma_audio_base_url).TrimEnd("/")
    Write-Host "Gemma: $(if(Endpoint-Ok "$gemmaBase/models"){'healthy'}else{'down'}) ($gemmaBase)"
    Write-Host "FasterQwen3TTS: $(if(Endpoint-Ok 'http://127.0.0.1:8881/health'){'healthy'}else{'down'}) ($ExpectedTtsModel)"
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
        if ($Component -ne "frontend") { Verify-Dependencies }
        foreach ($name in Selected-Components) { Start-One $name $state }
        if ($Open) { Start-Process "http://127.0.0.1:7862" }
    }
    "stop" {
        foreach ($name in Selected-Components-Reversed) { Stop-One $name $state }
    }
    "restart" {
        if ($Component -ne "frontend") { Verify-Dependencies }
        foreach ($name in Selected-Components-Reversed) { Stop-One $name $state }
        $state = Read-State
        foreach ($name in Selected-Components) { Start-One $name $state }
        if ($Open) { Start-Process "http://127.0.0.1:7862" }
    }
    "status" { Show-Status }
} }
