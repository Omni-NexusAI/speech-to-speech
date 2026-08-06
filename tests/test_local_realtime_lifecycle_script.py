import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "local_realtime.ps1"
FOREGROUND_SCRIPT = REPO_ROOT / "scripts" / "start_local_gemma_realtime_backend.ps1"
UI_SERVER = REPO_ROOT / "web" / "hf-realtime-voice" / "server.py"
TRANSIENT_CONTAINER_ID = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", re.IGNORECASE)
DOCKER_LIFECYCLE = re.compile(r"\bdocker\s+(?:inspect|start|stop|restart)\b", re.IGNORECASE)


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
    command = f'"{REPO_ROOT}\\.venv\\Scripts\\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 7862'
    expression = (
        f"$good=[pscustomobject]@{{commandLine='{command}'}}; "
        "$bad=[pscustomobject]@{commandLine='python -m http.server 7862'}; "
        "@((Test-Expected-Process 'frontend' $good),(Test-Expected-Process 'frontend' $bad)) | ConvertTo-Json -Compress"
    )
    assert _powershell(expression) == "[true,false]"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_script_contains_no_unsupported_select_object_reverse():
    assert "Select-Object -Reverse" not in SCRIPT.read_text(encoding="utf-8")


def test_managed_backend_uses_worktree_source_without_inheriting_pythonpath():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '$oldPythonPath = $env:PYTHONPATH' in source
    assert 'if ($Name -eq "backend") { Join-Path $RepoRoot "src" } else { "" }' in source
    assert '$env:PYTHONPATH = $oldPythonPath' in source


def test_managed_launcher_normalizes_duplicate_windows_path_keys():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '[Environment]::GetEnvironmentVariables("Process")' in source
    assert '$pathKeys.Count -gt 1' in source
    assert '[Environment]::SetEnvironmentVariable("Path", $pathValue, "Process")' in source


def test_backend_launchers_do_not_manage_models_or_use_transient_container_ids():
    for launcher in (SCRIPT, FOREGROUND_SCRIPT):
        source = launcher.read_text(encoding="utf-8")
        assert DOCKER_LIFECYCLE.search(source) is None, launcher
        assert TRANSIENT_CONTAINER_ID.search(source) is None, launcher

    foreground = FOREGROUND_SCRIPT.read_text(encoding="utf-8")
    assert "Invoke-RestMethod" not in foreground
    assert "Gemma and TTS services are user-managed" in foreground


def test_local_pipeline_reports_stable_tts_container_identity():
    source = UI_SERVER.read_text(encoding="utf-8")
    assert '"container": "qwen3-tts-faster"' in source
    assert TRANSIENT_CONTAINER_ID.search(source) is None
