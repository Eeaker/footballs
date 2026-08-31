from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAT_FILES = [
    "RUN_WINDOWS.bat", "INSTALL_WINDOWS.bat", "START_WINDOWS.bat", "CHECK_WINDOWS.bat",
    "DOWNLOAD_MODEL_WINDOWS.bat", "PREPARE_OFFLINE_WINDOWS.bat", "INSTALL_OFFLINE_WINDOWS.bat",
    "STOP_WINDOWS.bat", "REPAIR_WINDOWS.bat", "DIAGNOSE_WINDOWS.bat",
]
PS1_FILES = [
    "scripts/windows_install.ps1", "scripts/windows_prepare_offline.ps1", "scripts/windows_install_offline.ps1",
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

    installer = (ROOT / "scripts/windows_install.ps1").read_text(encoding="utf-8-sig")
    if 'print("Python"' in installer or "print('Python'" in installer:
        fail("installer contains the legacy PowerShell 5.1 native-quoting bug")
    if "--version" not in installer or "football_insight_install.mode" not in installer:
        fail("installer is missing robust version detection/completion marker")
    torch_probe_function = installer.split("function Test-Torch", 1)[-1].split("function Remove-Torch", 1)[0]
    if "| Out-Host" not in torch_probe_function or "$ProbeExitCode = $LASTEXITCODE" not in torch_probe_function:
        fail("installer Test-Torch leaks probe output into its boolean return value")
    torch_remove_function = installer.split("function Remove-Torch", 1)[-1].split("function Install-TorchPlan", 1)[0]
    if '$ErrorActionPreference = "Continue"' not in torch_remove_function or "$UninstallExitCode = $LASTEXITCODE" not in torch_remove_function:
        fail("installer Remove-Torch treats harmless pip stderr warnings as fatal")
    run = (ROOT / "RUN_WINDOWS.bat").read_text(encoding="ascii")
    if "football_insight_install.mode" not in run:
        fail("RUN_WINDOWS.bat does not require the completed-install marker")

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
