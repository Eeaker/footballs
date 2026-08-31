from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
import platform
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG_DIR = ROOT / "runtime" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
OUT = DIAG_DIR / "windows_torch_probe.json"


def load_dll(name: str) -> tuple[bool, str]:
    try:
        ctypes.WinDLL(name)
        return True, "loaded"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vc-only", action="store_true")
    ap.add_argument("--expect-cuda", action="store_true")
    args = ap.parse_args()

    data: dict[str, object] = {
        "python": sys.version.split()[0],
        "python_bits": struct.calcsize("P") * 8,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {k: pkg_version(k) for k in ("torch", "torchvision", "ultralytics", "easyocr")},
    }
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        data["nvidia_smi"] = {
            "returncode": proc.returncode,
            "output": proc.stdout.strip(),
            "error": proc.stderr.strip(),
        }
    except Exception as exc:
        data["nvidia_smi"] = {"error": f"{type(exc).__name__}: {exc}"}

    vc_names = ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "concrt140.dll"]
    vc = {name: {"ok": ok, "detail": detail} for name in vc_names for ok, detail in [load_dll(name)]}
    data["vc_runtime"] = vc
    vc_ok = all(bool(v["ok"]) for v in vc.values())
    data["vc_runtime_ok"] = vc_ok

    if args.vc_only:
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        for name, item in vc.items():
            print(("PASS" if item["ok"] else "FAIL"), name, item["detail"])
        return 0 if vc_ok else 10

    torch_info: dict[str, object] = {"import_ok": False}
    try:
        import torch
        torch_info.update({
            "import_ok": True,
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        })
        if torch.cuda.is_available():
            torch_info["device_name"] = torch.cuda.get_device_name(0)
            torch_info["device_capability"] = list(torch.cuda.get_device_capability(0))
            # Real CUDA smoke test, not just enumeration.
            a = torch.ones((64, 64), device="cuda")
            b = a @ a
            torch.cuda.synchronize()
            torch_info["cuda_smoke"] = float(b[0, 0].item()) == 64.0
        else:
            torch_info["cuda_smoke"] = False
    except Exception as exc:
        torch_info["error_type"] = type(exc).__name__
        torch_info["error"] = str(exc)
        # Try to give a more direct c10.dll diagnostic without importing torch.
        try:
            site = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
            c10 = site / "c10.dll"
            torch_info["c10_path"] = str(c10)
            torch_info["c10_exists"] = c10.is_file()
            if c10.is_file() and hasattr(os, "add_dll_directory"):
                with os.add_dll_directory(str(site)):
                    ctypes.WinDLL(str(c10))
                torch_info["c10_direct_load"] = "loaded"
        except Exception as dll_exc:
            torch_info["c10_direct_load"] = f"{type(dll_exc).__name__}: {dll_exc}"

    data["torch"] = torch_info
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    if not torch_info.get("import_ok"):
        return 20
    if args.expect_cuda and not torch_info.get("cuda_available"):
        return 30
    if args.expect_cuda and not torch_info.get("cuda_smoke"):
        return 31
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
