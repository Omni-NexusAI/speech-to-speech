import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
INSTALLER = SCRIPTS / "install-rts.ps1"
UNINSTALLER = SCRIPTS / "uninstall-rts.ps1"


def run_powershell(*args: str, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            *args,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
        encoding="utf-8",
        env=env,
    )


@pytest.mark.skipif(sys.platform != "win32", reason="rts shims are Windows launchers")
def test_repo_local_powershell_wrapper_forwards_arguments(tmp_path: Path):
    work = tmp_path / "scripts"
    work.mkdir()
    shutil.copy2(SCRIPTS / "rts.ps1", work / "rts.ps1")
    (work / "local_realtime.ps1").write_text("Write-Output ($args -join '|')\n", encoding="utf-8")

    result = run_powershell("-File", str(work / "rts.ps1"), "-Action", "status", "-Component", "all")

    assert result.stdout.strip() == "-Action|status|-Component|all"


@pytest.mark.skipif(sys.platform != "win32", reason="rts shims are Windows launchers")
def test_cmd_wrapper_forwards_arguments(tmp_path: Path):
    work = tmp_path / "scripts"
    work.mkdir()
    shutil.copy2(SCRIPTS / "rts.ps1", work / "rts.ps1")
    shutil.copy2(SCRIPTS / "rts.cmd", work / "rts.cmd")
    (work / "local_realtime.ps1").write_text("Write-Output ($args -join '|')\n", encoding="utf-8")

    result = subprocess.run(
        ["cmd.exe", "/c", str(work / "rts.cmd"), "-Action", "status", "-Component", "all"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "-Action|status|-Component|all"


@pytest.mark.skipif(sys.platform != "win32", reason="rts installer is Windows-specific")
def test_installer_creates_global_shims_with_sidecar_target(tmp_path: Path):
    install_dir = tmp_path / "bin"
    profile = tmp_path / "profile.ps1"
    target = tmp_path / "local_realtime.ps1"
    target.write_text("Write-Output ($args -join '|')\n", encoding="utf-8")

    run_powershell(
        "-File",
        str(INSTALLER),
        "-InstallDir",
        str(install_dir),
        "-TargetScript",
        str(target),
        "-PathScope",
        "None",
        "-ProfilePath",
        str(profile),
    )

    assert (install_dir / "rts.ps1").is_file()
    assert (install_dir / "rts.cmd").is_file()
    assert (install_dir / "rts.target").read_text(encoding="utf-8-sig").strip() == str(target)
    profile_text = profile.read_text(encoding="utf-8")
    assert "# BEGIN speech-to-speech rts" in profile_text
    assert str(install_dir / "rts.cmd") in profile_text

    result = run_powershell("-File", str(install_dir / "rts.ps1"), "-Action", "status")
    assert result.stdout.strip() == "-Action|status"


@pytest.mark.skipif(sys.platform != "win32", reason="rts installer is Windows-specific")
def test_installed_sidecar_launches_target_from_unicode_path(tmp_path: Path):
    install_dir = tmp_path / "bin"
    target_dir = tmp_path / "配置"
    target_dir.mkdir()
    target = target_dir / "local_realtime.ps1"
    target.write_text("Write-Output ($args -join '|')\n", encoding="utf-8")

    run_powershell(
        "-File",
        str(INSTALLER),
        "-InstallDir",
        str(install_dir),
        "-TargetScript",
        str(target),
        "-PathScope",
        "None",
        "-SkipProfileFunction",
    )

    assert (install_dir / "rts.target").read_text(encoding="utf-8-sig").strip() == str(target)
    result = run_powershell("-File", str(install_dir / "rts.ps1"), "-Action", "status")
    assert result.stdout.strip() == "-Action|status"


@pytest.mark.skipif(sys.platform != "win32", reason="rts installer is Windows-specific")
def test_installer_is_idempotent_and_deduplicates_process_path(tmp_path: Path):
    install_dir = tmp_path / "bin"
    profile = tmp_path / "profile.ps1"
    target = tmp_path / "local_realtime.ps1"
    target.write_text("Write-Output ok\n", encoding="utf-8")

    result = run_powershell(
        "-Command",
        f"$env:Path='{tmp_path / 'other'}'; "
        f"& '{INSTALLER}' -InstallDir '{install_dir}' -TargetScript '{target}' -PathScope Process -ProfilePath '{profile}'; "
        f"& '{INSTALLER}' -InstallDir '{install_dir}' -TargetScript '{target}' -PathScope Process -ProfilePath '{profile}'; "
        "$env:Path -split ';' | Where-Object { $_ -eq '"
        + str(install_dir).replace("'", "''")
        + "' } | Measure-Object | Select-Object -ExpandProperty Count",
    )

    assert result.stdout.strip().splitlines()[-1] == "1"
    profile_text = profile.read_text(encoding="utf-8")
    assert profile_text.count("# BEGIN speech-to-speech rts") == 1


@pytest.mark.skipif(sys.platform != "win32", reason="rts installer is Windows-specific")
def test_installed_profile_function_invokes_cmd_shim(tmp_path: Path):
    install_dir = tmp_path / "bin"
    profile = tmp_path / "profile.ps1"
    target = tmp_path / "local_realtime.ps1"
    target.write_text("Write-Output ($args -join '|')\n", encoding="utf-8")

    run_powershell(
        "-File",
        str(INSTALLER),
        "-InstallDir",
        str(install_dir),
        "-TargetScript",
        str(target),
        "-PathScope",
        "None",
        "-ProfilePath",
        str(profile),
    )

    result = run_powershell(
        "-Command",
        f". '{profile}'; rts -Action status -Component all",
        cwd=tmp_path,
    )

    assert result.stdout.strip() == "-Action|status|-Component|all"


@pytest.mark.skipif(sys.platform != "win32", reason="rts uninstaller is Windows-specific")
def test_uninstaller_removes_shims_and_optional_process_path(tmp_path: Path):
    install_dir = tmp_path / "bin"
    profile = tmp_path / "profile.ps1"
    target = tmp_path / "local_realtime.ps1"
    target.write_text("Write-Output ok\n", encoding="utf-8")

    run_powershell(
        "-File",
        str(INSTALLER),
        "-InstallDir",
        str(install_dir),
        "-TargetScript",
        str(target),
        "-PathScope",
        "None",
        "-ProfilePath",
        str(profile),
    )

    result = run_powershell(
        "-Command",
        f"$env:Path='{tmp_path / 'other'};{install_dir}'; "
        f"& '{UNINSTALLER}' -InstallDir '{install_dir}' -PathScope Process -RemovePath -ProfilePath '{profile}'; "
        "$files=@('rts.ps1','rts.cmd','rts.target') | ForEach-Object { Test-Path (Join-Path '"
        + str(install_dir).replace("'", "''")
        + "' $_) }; "
        "[string]::Join(',', $files) + ';path=' + $env:Path",
    )

    assert "False,False,False" in result.stdout
    assert str(install_dir) not in result.stdout.rsplit(";path=", 1)[-1]
    assert "# BEGIN speech-to-speech rts" not in profile.read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform != "win32", reason="rts installer is Windows-specific")
def test_profile_install_uninstall_preserves_unicode_and_original_text(tmp_path: Path):
    install_dir = tmp_path / "bin"
    profile = tmp_path / "profile.ps1"
    target = tmp_path / "local_realtime.ps1"
    original = "# Existing profile\r\nWrite-Output 'Grüße 世界'\r\n"
    profile.write_text(original, encoding="utf-8-sig", newline="")
    target.write_text("Write-Output ok\n", encoding="utf-8")

    run_powershell(
        "-File",
        str(INSTALLER),
        "-InstallDir",
        str(install_dir),
        "-TargetScript",
        str(target),
        "-PathScope",
        "None",
        "-ProfilePath",
        str(profile),
    )
    installed = profile.read_text(encoding="utf-8-sig")
    assert "Grüße 世界" in installed
    assert "# BEGIN speech-to-speech rts" in installed

    run_powershell(
        "-File",
        str(UNINSTALLER),
        "-InstallDir",
        str(install_dir),
        "-PathScope",
        "None",
        "-ProfilePath",
        str(profile),
    )
    assert profile.read_text(encoding="utf-8-sig", newline="") == original


@pytest.mark.skipif(sys.platform != "win32", reason="rts installer is Windows-specific")
def test_user_path_update_preserves_unrelated_effective_process_entries(tmp_path: Path):
    shim = str(tmp_path / "shim")
    escaped_shim = shim.replace("'", "''")
    expression = (
        f". '{INSTALLER}' -InternalNoRun; "
        "$script:userPath='USER_ONLY'; "
        "function Get-PathValue([string]$Scope) { if($Scope -eq 'Process'){ return [string]$env:Path }; return $script:userPath }; "
        "function Set-PathValue([string]$Scope,[string]$Value) { if($Scope -eq 'Process'){ $env:Path=$Value } else { $script:userPath=$Value } }; "
        "$env:Path='PROCESS_ONLY;MACHINE_ONLY'; "
        f"$null=Add-PathEntry 'User' '{escaped_shim}'; "
        "Write-Output ('user=' + $script:userPath); Write-Output ('process=' + $env:Path); "
        f". '{UNINSTALLER}' -InternalNoRun; "
        "function Get-PathValue([string]$Scope) { if($Scope -eq 'Process'){ return [string]$env:Path }; return $script:userPath }; "
        "function Set-PathValue([string]$Scope,[string]$Value) { if($Scope -eq 'Process'){ $env:Path=$Value } else { $script:userPath=$Value } }; "
        f"$null=Remove-PathEntry 'User' '{escaped_shim}'; "
        "Write-Output ('user-after=' + $script:userPath); Write-Output ('process-after=' + $env:Path)"
    )
    result = run_powershell("-Command", expression)
    output = result.stdout.strip().splitlines()
    assert output[0] == f"user=USER_ONLY;{shim}"
    assert output[1] == f"process=PROCESS_ONLY;MACHINE_ONLY;{shim}"
    assert output[2] == "user-after=USER_ONLY"
    assert output[3] == "process-after=PROCESS_ONLY;MACHINE_ONLY"
