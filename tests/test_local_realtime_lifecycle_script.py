import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "local_realtime.ps1"
FRONTEND_SCRIPT = REPO_ROOT / "scripts" / "start_hf_realtime_frontend.ps1"


def _powershell(expression: str) -> str:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            f". '{SCRIPT}' -InternalNoRun; {expression}",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_component_shutdown_order_is_powershell_51_compatible():
    assert _powershell("$Component='all'; (Selected-Components-Reversed) -join ','") == "frontend,backend"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_exact_repo_commands_are_recognized_but_unknown_commands_are_not():
    command = (
        f'"{REPO_ROOT}\\.venv\\Scripts\\python.exe" -m uvicorn server:app '
        f'--app-dir {REPO_ROOT}\\web\\hf-realtime-voice --host 127.0.0.1 --port 7862'
    )
    expression = (
        f"$good=[pscustomobject]@{{commandLine='{command}'}}; "
        "$bad=[pscustomobject]@{commandLine='python -m http.server 7862'}; "
        "@((Test-Expected-Process 'frontend' $good),(Test-Expected-Process 'frontend' $bad)) | ConvertTo-Json -Compress"
    )
    assert _powershell(expression) == "[true,false]"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_recorded_listener_uses_roundtrip_utc_identity_after_json_reload():
    expression = (
        "$record=('{\"listenerPid\":42,\"listenerStartTimeUtc\":' + "
        "'\"2026-08-26T08:27:45.40128Z\",\"listenerExecutable\":' + "
        "'\"python.exe\"}' | ConvertFrom-Json); "
        "$listener=[pscustomobject]@{pid=42;"
        "startTimeUtc='2026-08-26T08:27:45.4012800Z';"
        "executable='python.exe'}; "
        "Test-Recorded-Listener $record $listener"
    )
    assert _powershell(expression) == "True"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_direct_venv_child_is_owned_but_an_unrelated_listener_is_rejected():
    command = (
        f'"{REPO_ROOT}\\.venv\\Scripts\\python.exe" -m uvicorn server:app '
        f'--app-dir {REPO_ROOT}\\web\\hf-realtime-voice --host 127.0.0.1 --port 7862'
    )
    expression = (
        "$launcher=[pscustomobject]@{pid=42;startTimeUtc='2026-09-03T23:00:00.0000000Z';"
        f"executable='python.exe';commandLine='{command}'}}; "
        "$child=[pscustomobject]@{pid=43;parentPid=42;startTimeUtc='2026-09-03T23:00:00.1000000Z';"
        f"executable='python.exe';commandLine='{command}'}}; "
        "$unrelated=[pscustomobject]@{pid=44;parentPid=999;startTimeUtc='2026-09-03T23:00:00.1000000Z';"
        f"executable='python.exe';commandLine='{command}'}}; "
        "@((Test-Launched-Listener 'frontend' $launcher $child),"
        "(Test-Launched-Listener 'frontend' $launcher $unrelated)) | ConvertTo-Json -Compress"
    )
    assert _powershell(expression) == "[true,false]"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_script_contains_no_unsupported_select_object_reverse():
    assert "Select-Object -Reverse" not in SCRIPT.read_text(encoding="utf-8")


def test_managed_backend_uses_worktree_source_without_inheriting_pythonpath():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "function Test-PythonRuntime" in source
    assert "function Resolve-PythonRuntime" in source
    assert "import nltk, requests, torch, transformers, uvicorn" in source
    assert "import speech_to_speech.s2s_pipeline" in source
    assert "Test-PythonRuntime $candidate $Name" in source
    assert '$oldPythonPath = $env:PYTHONPATH' in source
    assert '$oldErrorActionPreference = $ErrorActionPreference' in source
    assert '$ErrorActionPreference = "Continue"' in source
    assert 'if ($Name -eq "backend") { Join-Path $RepoRoot "src" } else { "" }' in source
    assert '$env:PYTHONPATH = $oldPythonPath' in source
    assert '$ErrorActionPreference = $oldErrorActionPreference' in source


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_runtime_resolution_is_component_scoped_and_prefers_a_complete_candidate():
    expression = (
        "$WorktreePython='worktree-python'; $SharedPython='shared-python'; "
        "function Test-PythonRuntime([string]$Candidate,[string]$Name) { "
        "if($Name -eq 'frontend'){ return $Candidate -eq 'worktree-python' }; "
        "return $Candidate -eq 'shared-python' }; "
        "@((Resolve-PythonRuntime 'backend'),(Resolve-PythonRuntime 'frontend')) | ConvertTo-Json -Compress"
    )
    assert _powershell(expression) == '["shared-python","worktree-python"]'


def test_managed_frontend_uses_the_selected_persisted_gemma_identity():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '$FrontendRoot = Join-Path $RepoRoot "web\\hf-realtime-voice"' in source
    assert '"--app-dir", $FrontendRoot' in source
    assert "$gemma = Resolve-GemmaRuntimeIdentity" in source
    assert '$env:GEMMA_AUDIO_BASE_URL = [string]$gemma.baseUrl' in source
    assert '$env:GEMMA_AUDIO_MODEL = [string]$gemma.model' in source
    assert '$env:GEMMA_AUDIO_BASE_URL = $oldGemmaBaseUrl' in source
    assert '$env:GEMMA_AUDIO_MODEL = $oldGemmaModel' in source


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_remote_gemma_identity_is_resolved_from_public_ui_settings(tmp_path):
    settings_path = tmp_path / "hf_realtime_ui_settings.json"
    settings_path.write_text(
        '{"modelProvider":"remote","modelUrl":"http://192.168.0.178:8081/v1/",'
        '"modelName":"gemma-4-12b"}',
        encoding="utf-8",
    )
    escaped = str(settings_path).replace("'", "''")
    output = _powershell(
        f"Resolve-GemmaRuntimeIdentity '{escaped}' | ConvertTo-Json -Compress"
    )

    assert output == (
        '{"provider":"remote","baseUrl":"http://192.168.0.178:8081/v1",'
        '"model":"gemma-4-12b"}'
    )


def test_foreground_frontend_uses_and_restores_the_configured_gemma_identity():
    source = FRONTEND_SCRIPT.read_text(encoding="utf-8")
    assert 'examples\\local_gemma_fasterqwen3tts.json' in source
    assert '$env:GEMMA_AUDIO_BASE_URL = [string]$runtimeConfig.gemma_audio_base_url' in source
    assert '$env:GEMMA_AUDIO_MODEL = [string]$runtimeConfig.gemma_audio_model_name' in source
    assert '$env:GEMMA_AUDIO_BASE_URL = $oldGemmaBaseUrl' in source
    assert '$env:GEMMA_AUDIO_MODEL = $oldGemmaModel' in source


def test_managed_launcher_normalizes_duplicate_windows_path_keys():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '[Environment]::GetEnvironmentVariables("Process")' in source
    assert '$pathKeys.Count -gt 1' in source
    assert '[Environment]::SetEnvironmentVariable("Path", $pathValue, "Process")' in source


def test_managed_launcher_snapshots_identity_before_a_venv_shim_can_exit():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "$launcherInfo = [pscustomobject]@{" in source
    assert "Test-Launched-Listener $Name $launcherInfo $listener" in source


def test_owned_process_stop_has_a_powershell_51_process_object_fallback():
    source = SCRIPT.read_text(encoding="utf-8")
    stop_block = source.split("function Stop-Known-Process", 1)[1].split("function Stop-One", 1)[0]
    assert "Stop-Process -InputObject $process -ErrorAction Stop" in stop_block
    assert stop_block.count("Get-Process -Id $ProcessId") == 1
    assert "$process.HasExited" in stop_block
    assert "$process.Kill()" in stop_block
    assert "$process.WaitForExit(5000)" in stop_block
    assert "$remaining" not in stop_block


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_owned_process_stop_treats_a_raced_normal_exit_as_success_without_pid_rebind():
    expression = (
        "$exe=(Get-Process -Id $PID).Path; "
        "$child=Start-Process -FilePath $exe -ArgumentList '-NoProfile','-Command',"
        "'Start-Sleep -Seconds 30' -WindowStyle Hidden -PassThru; "
        "function Stop-Process { [CmdletBinding()] param("
        "[Parameter(ValueFromPipeline=$true)]$InputObject,[switch]$Force); "
        "$InputObject.Kill(); $InputObject.WaitForExit(); throw 'simulated post-exit error' }; "
        "try { Stop-Known-Process $child.Id; $child.Refresh(); [bool]$child.HasExited } "
        "finally { try { if(-not $child.HasExited){$child.Kill()} } catch {} }"
    )
    assert _powershell(expression).splitlines()[-1] == "True"


def test_backend_only_start_and_restart_report_a_healthy_existing_ui_without_opening_it():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "function Write-FrontendUrlIfHealthy" in source
    assert 'if ($Component -eq "backend") { Write-FrontendUrlIfHealthy | Out-Null }' in source
    assert "Only -Open is allowed to launch a browser" in source


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_verified_frontend_url_helper_never_opens_a_browser():
    expression = (
        "function Endpoint-Ok([string]$Url) { return $true }; "
        "function Start-Process([string]$Url) { Write-Output ('OPEN:' + $Url) }; "
        "Write-FrontendUrlIfHealthy | Out-Null"
    )
    output = _powershell(expression)
    assert "HF Realtime Voice UI: http://127.0.0.1:7862" in output
    assert "OPEN:" not in output


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_open_refuses_an_unverified_frontend_without_launching_a_browser():
    expression = (
        "function Endpoint-Ok([string]$Url) { return $false }; "
        "function Start-Process([string]$Url) { Write-Output ('OPEN:' + $Url) }; "
        "try { Open-FrontendIfReady } catch { Write-Output $_.Exception.Message }"
    )
    output = _powershell(expression)
    assert "Cannot open HF Realtime Voice UI" in output
    assert "OPEN:" not in output


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_open_launches_only_after_frontend_health_is_verified():
    expression = (
        "function Endpoint-Ok([string]$Url) { return $true }; "
        "function Start-Process([string]$Url) { Write-Output ('OPEN:' + $Url) }; "
        "Open-FrontendIfReady"
    )
    assert _powershell(expression) == "OPEN:http://127.0.0.1:7862"


def test_status_does_not_claim_a_fixed_turn_cap_or_disabled_compaction():
    text = (Path(__file__).parents[1] / "scripts" / "local_realtime.ps1").read_text(encoding="utf-8")
    assert "Context: 30 complete turns" not in text
    assert "compact_history=false" not in text
    assert "active compaction policy and budget" in text
