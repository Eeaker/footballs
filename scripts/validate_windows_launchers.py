from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAT_FILES = [
    "DEPLOY_ONE_CLICK_WINDOWS.bat", "DEPLOY_DOCKER_WINDOWS.bat",
    "CHECK_WINDOWS.bat", "DOWNLOAD_MODEL_WINDOWS.bat",
    "PREPARE_OFFLINE_WINDOWS.bat", "INSTALL_OFFLINE_WINDOWS.bat",
    "STOP_WINDOWS.bat", "REPAIR_WINDOWS.bat", "DIAGNOSE_WINDOWS.bat",
]
REMOVED_LAUNCHERS = ["RUN_WINDOWS.bat", "INSTALL_WINDOWS.bat", "START_WINDOWS.bat"]
PS1_FILES = [
    "deploy.ps1",
    "scripts/windows_install.ps1",
    "scripts/windows_prepare_offline.ps1",
    "scripts/windows_install_offline.ps1",
]


def fail(message: str) -> None:
    raise SystemExit("FAIL: " + message)


def main() -> None:
    for rel in BAT_FILES:
        p = ROOT / rel
        if not p.is_file():
            fail(f"missing {rel}")
        data = p.read_bytes()
        if any(b >= 128 for b in data):
            fail(f"{rel} is not ASCII-only")
        if b"\r\n" not in data or b"\n" in data.replace(b"\r\n", b""):
            fail(f"{rel} is not CRLF-only")
        text = data.decode("ascii")
        if 'cd /d "%~dp0"' not in text:
            fail(f"{rel} does not anchor working directory safely")

    vbs = ROOT / "PRESENT_WINDOWS.vbs"
    if not vbs.is_file() or any(b >= 128 for b in vbs.read_bytes()):
        fail("PRESENT_WINDOWS.vbs must be ASCII-only")

    for rel in PS1_FILES:
        p = ROOT / rel
        if not p.is_file():
            fail(f"missing {rel}")
        data = p.read_bytes()
        if not data.startswith(b"\xef\xbb\xbf"):
            fail(f"{rel} must use UTF-8 BOM for Windows PowerShell 5.1")

    deploy = (ROOT / "deploy.ps1").read_text(encoding="utf-8-sig")
    if "docker info *> $null" in deploy:
        fail("deploy.ps1 treats an inactive Docker engine as a terminating PowerShell error")
    required_bootstrap_markers = (
        "function Test-DockerEngine",
        "function Start-DockerDesktop",
        "docker info 1>nul 2>nul",
        "Docker Desktop 启动超时",
    )
    if any(marker not in deploy for marker in required_bootstrap_markers):
        fail("deploy.ps1 is missing automatic Docker Desktop startup/readiness handling")

    installer = (ROOT / "scripts/windows_install.ps1").read_text(encoding="utf-8-sig")
    if 'print("Python"' in installer or "print('Python'" in installer:
        fail("installer contains the legacy PowerShell 5.1 native-quoting bug")
    if "--version" not in installer or "football_insight_install.mode" not in installer:
        fail("installer is missing robust version detection/completion marker")
    python_bootstrap_markers = (
        "function Install-PrivatePython",
        "function Ensure-Pip",
        "function Get-PrivatePython",
        "runtime\\python",
        "python-3.12.10-amd64.exe",
        "67B5635E80EA51072B87941312D00EC8927C4DB9BA18938F7AD2D27B328B95FB",
        "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe",
        "ensurepip",
        "Install-PrivatePython",
    )
    if any(marker not in installer for marker in python_bootstrap_markers):
        fail("installer is missing private Python 3.12 / pip bootstrap from zero")
    if "Get-Command python.exe" in installer.split("function Get-OfficialFile", 1)[0]:
        fail("one-click installer still searches for a preinstalled system Python")
    if 'Fail "未找到 64 位 Python 3.11/3.12。"' in installer:
        fail("installer still requires a preinstalled system Python")
    for rel in REMOVED_LAUNCHERS:
        if (ROOT / rel).exists():
            fail(f"{rel} is redundant and must be removed; use DEPLOY_ONE_CLICK_WINDOWS.bat only")
    offline_install = (ROOT / "scripts/windows_install_offline.ps1").read_text(encoding="utf-8-sig")
    if "Install-PrivatePythonFromWheelhouse" not in offline_install or "python-3.12.10-amd64.exe" not in offline_install:
        fail("offline installer cannot bootstrap Python from wheelhouse")
    if "Ensure-Pip" not in offline_install:
        fail("offline installer does not bootstrap pip with ensurepip")
    prepare_offline = (ROOT / "scripts/windows_prepare_offline.ps1").read_text(encoding="utf-8-sig")
    if "python-3.12.10-amd64.exe" not in prepare_offline or "67B5635E80EA51072B87941312D00EC8927C4DB9BA18938F7AD2D27B328B95FB" not in prepare_offline:
        fail("offline prepare script does not cache the official Python installer")
    torch_probe_function = installer.split("function Test-Torch", 1)[-1].split("function Remove-Torch", 1)[0]
    if "| Out-Host" not in torch_probe_function or "$ProbeExitCode = $LASTEXITCODE" not in torch_probe_function:
        fail("installer Test-Torch leaks probe output into its boolean return value")
    torch_remove_function = installer.split("function Remove-Torch", 1)[-1].split("function Install-TorchPlan", 1)[0]
    if '$ErrorActionPreference = "Continue"' not in torch_remove_function or "$UninstallExitCode = $LASTEXITCODE" not in torch_remove_function:
        fail("installer Remove-Torch treats harmless pip stderr warnings as fatal")
    one_click = (ROOT / "DEPLOY_ONE_CLICK_WINDOWS.bat").read_text(encoding="ascii")
    if "windows_install.ps1" not in one_click or "-Mode AUTO" not in one_click:
        fail("DEPLOY_ONE_CLICK_WINDOWS.bat is not a silent AUTO installer")
    if "windows_launcher.py" not in one_click:
        fail("DEPLOY_ONE_CLICK_WINDOWS.bat does not start the app itself")
    if "RUN_WINDOWS.bat" in one_click or "INSTALL_WINDOWS.bat" in one_click or "START_WINDOWS.bat" in one_click:
        fail("DEPLOY_ONE_CLICK_WINDOWS.bat must not call redundant launchers")
    if "deploy.ps1" in one_click:
        fail("DEPLOY_ONE_CLICK_WINDOWS.bat must be native online deploy, not Docker")
    if "set /p" in one_click:
        fail("DEPLOY_ONE_CLICK_WINDOWS.bat still prompts instead of deploying automatically")
    if "football_insight_install.mode" not in one_click:
        fail("DEPLOY_ONE_CLICK_WINDOWS.bat does not require the completed-install marker")
    if "自动回退到 CPU" not in installer:
        fail("installer does not fall back from AUTO GPU to CPU")

    # Exercise the launcher with the same module path shape as `python scripts/windows_launcher.py`.
    # Stub uvicorn so this remains a dependency-free packaging check and does not start a server.
    launcher = ROOT / "scripts" / "windows_launcher.py"
    probe_code = "\n".join([
        "import runpy, sys, types",
        f"root = {str(ROOT)!r}",
        f"scripts = {str(ROOT / 'scripts')!r}",
        f"launcher = {str(launcher)!r}",
        "sys.modules['uvicorn'] = types.ModuleType('uvicorn')",
        "sys.path[:] = [p for p in sys.path if p and p.casefold() != root.casefold()]",
        "sys.path.insert(0, scripts)",
        "runpy.run_path(launcher, run_name='windows_launcher_validation')",
    ])
    launcher_proc = subprocess.run(
        [sys.executable, "-c", probe_code],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
    )
    if launcher_proc.returncode != 0:
        fail("windows_launcher.py cannot import the app when executed as a script: " + launcher_proc.stderr.strip())

    print("Windows launcher encoding validation: PASS")
    print(f"  BAT ASCII+CRLF: {len(BAT_FILES)}/{len(BAT_FILES)}")
    print("  VBS ASCII: PASS")
    print(f"  PowerShell UTF-8 BOM: {len(PS1_FILES)}/{len(PS1_FILES)}")
    print("  Script-mode app import: PASS")


if __name__ == "__main__":
    main()
