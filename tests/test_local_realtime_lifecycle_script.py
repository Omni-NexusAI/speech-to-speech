import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "local_realtime.ps1"


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
