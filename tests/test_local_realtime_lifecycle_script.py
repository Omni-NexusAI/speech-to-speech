import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "local_realtime.ps1"
SCRIPTS_CONTRACT = REPO_ROOT / "scripts" / "AGENTS.md"
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


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _source_identity_for(repo: Path) -> dict:
    escaped = str(repo).replace("'", "''")
    payload = _powershell(
        f"$RepoRoot='{escaped}'; $identity=Get-SourceIdentity; "
        "[pscustomobject]@{revision=$identity.revision;dirty=$identity.dirty;"
        "fingerprint=$identity.fingerprint;assets=$identity.uiAssetGeneration} | ConvertTo-Json -Compress"
    )
    return json.loads(payload)


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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_managed_launcher_builds_content_free_source_identity():
    payload = _powershell(
        "$identity=Get-SourceIdentity; "
        "[pscustomobject]@{revision=$identity.revision;dirty=$identity.dirty;"
        "fingerprint=$identity.fingerprint;assets=$identity.uiAssetGeneration} | ConvertTo-Json -Compress"
    )
    identity = json.loads(payload)

    assert re.fullmatch(r"[0-9a-f]{40}", identity["revision"])
    assert isinstance(identity["dirty"], bool)
    assert re.fullmatch(r"[0-9a-f]{64}", identity["fingerprint"])
    assert identity["assets"].startswith("main=")
    assert ";ws=" in identity["assets"]
    assert ";chat=" in identity["assets"]
    assert ";playback=" in identity["assets"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_source_identity_comparison_rejects_any_mismatch():
    revision = "a" * 40
    fingerprint = "b" * 64
    assets = "main=31;ws=18;chat=3;playback=16"
    expression = (
        f"$expected=[pscustomobject]@{{revision='{revision}';dirty=$false;fingerprint='{fingerprint}';"
        f"uiAssetGeneration='{assets}'}}; "
        f"$same=[pscustomobject]@{{source_revision='{revision}';source_dirty=$false;"
        f"source_fingerprint='{fingerprint}';ui_asset_generation='{assets}'}}; "
        f"$stale=[pscustomobject]@{{source_revision='{revision}';source_dirty=$false;"
        f"source_fingerprint='{'c' * 64}';ui_asset_generation='{assets}'}}; "
        f"$unknown=[pscustomobject]@{{source_revision='unknown';source_dirty=$false;"
        f"source_fingerprint='{fingerprint}';ui_asset_generation='{assets}'}}; "
        f"$untyped=[pscustomobject]@{{source_revision='{revision}';source_dirty='false';"
        f"source_fingerprint='{fingerprint}';ui_asset_generation='{assets}'}}; "
        f"$badAssets=[pscustomobject]@{{source_revision='{revision}';source_dirty=$false;"
        f"source_fingerprint='{fingerprint}';ui_asset_generation='unknown'}}; "
        f"$caseMismatch=[pscustomobject]@{{source_revision='{'A' * 40}';source_dirty=$false;"
        f"source_fingerprint='{fingerprint}';ui_asset_generation='{assets}'}}; "
        "@((Test-SourceIdentityMatch $same $expected),(Test-SourceIdentityMatch $stale $expected),"
        "(Test-SourceIdentityMatch $unknown $expected),(Test-SourceIdentityMatch $untyped $expected),"
        "(Test-SourceIdentityMatch $badAssets $expected),(Test-SourceIdentityMatch $caseMismatch $expected)) "
        "| ConvertTo-Json -Compress"
    )
    assert _powershell(expression) == "[true,false,false,false,false,false]"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_managed_mutations_fail_before_stop_or_start_when_source_identity_is_invalid():
    expression = (
        "$invalid=[pscustomobject]@{revision='unknown';dirty=$true;fingerprint='unknown';"
        "uiAssetGeneration='unknown'}; "
        "try { Assert-ValidSourceIdentity $invalid; 'missed' } catch { $_.Exception.Message }"
    )
    assert _powershell(expression) == "Runtime source identity is incomplete; refusing managed start."

    source = SCRIPT.read_text(encoding="utf-8")
    start_case = source.split('"start" {', maxsplit=1)[1].split('"stop" {', maxsplit=1)[0]
    restart_case = source.split('"restart" {', maxsplit=1)[1].split('"status" {', maxsplit=1)[0]
    assert start_case.index("Assert-ValidSourceIdentity") < start_case.index("Start-One")
    assert restart_case.index("Assert-ValidSourceIdentity") < restart_case.index("Stop-One")
    assert restart_case.index("Assert-ValidSourceIdentity") < restart_case.index("Start-One")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_later_started_unknown_listener_is_never_treated_as_launcher_owned():
    expression = (
        "$launcher=[pscustomobject]@{pid=41001;parentPid=0;startTimeUtc='2026-08-12T20:00:00Z';"
        "commandLine='python known-launcher'}; "
        "$unknown=[pscustomobject]@{pid=41002;parentPid=0;startTimeUtc='2026-08-12T20:00:01Z';"
        "commandLine='python -m http.server 7862'}; "
        "(Test-Started-Listener-Ownership 'frontend' $unknown $launcher) | ConvertTo-Json -Compress"
    )
    assert _powershell(expression) == "false"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_independent_matching_command_listener_is_not_launcher_owned():
    expression = (
        "$launcher=[pscustomobject]@{pid=42001;parentPid=0;startTimeUtc='2026-08-12T20:00:00Z';"
        "commandLine='python -m uvicorn server:app --host 127.0.0.1 --port 7862'}; "
        "$independent=[pscustomobject]@{pid=42002;parentPid=0;startTimeUtc='2026-08-12T20:00:01Z';"
        "commandLine='python -m uvicorn server:app --host 127.0.0.1 --port 7862'}; "
        "(Test-Started-Listener-Ownership 'frontend' $independent $launcher) | ConvertTo-Json -Compress"
    )
    assert _powershell(expression) == "false"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_source_identity_excludes_docs_but_tracks_runtime_changes(tmp_path: Path):
    repo = tmp_path / "identity-fixture"
    (repo / "src").mkdir(parents=True)
    (repo / "web" / "hf-realtime-voice").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "web" / "hf-realtime-voice" / "main.js").write_text("export const value = 1;\n", encoding="utf-8")
    (repo / "README.md").write_text("initial docs\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "identity@example.invalid")
    _git(repo, "config", "user.name", "Runtime Identity Test")
    _git(repo, "add", "--", "src/app.py", "web/hf-realtime-voice/main.js", "README.md")
    _git(repo, "commit", "-m", "fixture")

    baseline = _source_identity_for(repo)
    (repo / "README.md").write_text("doc-only edit\n", encoding="utf-8")
    docs_only = _source_identity_for(repo)
    assert docs_only["dirty"] is False
    assert docs_only["fingerprint"] == baseline["fingerprint"]

    _git(repo, "add", "--", "README.md")
    _git(repo, "commit", "-m", "docs only")
    docs_commit = _source_identity_for(repo)
    assert docs_commit["revision"] != baseline["revision"]
    assert docs_commit["dirty"] is False
    assert docs_commit["fingerprint"] == baseline["fingerprint"]

    (repo / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    runtime_change = _source_identity_for(repo)
    assert runtime_change["dirty"] is True
    assert runtime_change["fingerprint"] != baseline["fingerprint"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell lifecycle script")
def test_source_identity_fails_closed_when_runtime_enumeration_is_empty(tmp_path: Path):
    repo = tmp_path / "empty-runtime-fixture"
    repo.mkdir()
    (repo / "README.md").write_text("docs only\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "identity@example.invalid")
    _git(repo, "config", "user.name", "Runtime Identity Test")
    _git(repo, "add", "--", "README.md")
    _git(repo, "commit", "-m", "docs only")

    identity = _source_identity_for(repo)
    assert identity["fingerprint"] == "unknown"


def test_managed_launcher_injects_and_restores_source_identity_environment():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "$StateVersion = 3" in source
    for name in (
        "S2S_RUNTIME_REVISION",
        "S2S_RUNTIME_DIRTY",
        "S2S_RUNTIME_SOURCE_FINGERPRINT",
        "S2S_UI_ASSET_GENERATION",
    ):
        assert source.count(f"$env:{name}") >= 3
    assert "STALE SOURCE" in source
    assert "Refusing adoption" in source
    assert "Runtime source identity is incomplete; refusing managed start." in source
    assert "Test-Started-Listener-Ownership $Name $listener $launcherInfo" in source
    assert "Refusing ownership without stopping that listener" in source
    assert "$startedAfterLauncher" not in source


def test_backend_launchers_do_not_manage_models_or_use_transient_container_ids():
    for launcher in (SCRIPT, FOREGROUND_SCRIPT):
        source = launcher.read_text(encoding="utf-8")
        assert DOCKER_LIFECYCLE.search(source) is None, launcher
        assert TRANSIENT_CONTAINER_ID.search(source) is None, launcher

    foreground = FOREGROUND_SCRIPT.read_text(encoding="utf-8")
    assert "Invoke-RestMethod" not in foreground
    assert "Gemma and TTS services are user-managed" in foreground


def test_installed_target_contract_keeps_rollback_record_passive():
    contract = SCRIPTS_CONTRACT.read_text(encoding="utf-8")
    assert "reads only `rts.target`" in contract
    assert "`rts.target.rollback`" in contract
    assert "never execute or overwrite that record implicitly" in contract
    assert "rts.target.rollback" not in SCRIPT.read_text(encoding="utf-8")


def test_local_pipeline_reports_stable_tts_container_identity():
    source = UI_SERVER.read_text(encoding="utf-8")
    assert '"container": "qwen3-tts-faster"' in source
    assert TRANSIENT_CONTAINER_ID.search(source) is None
