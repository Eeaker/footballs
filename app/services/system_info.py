from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from app.config import SYSTEM_ROOT, MODELS_ROOT, RUNTIME_ROOT, SYSTEM_VERSION


def _portable_ffmpeg() -> str | None:
    direct = shutil.which("ffmpeg")
    if direct:
        return direct
    try:
        import imageio_ffmpeg
        candidate = imageio_ffmpeg.get_ffmpeg_exe()
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    except Exception:
        pass
    return None


def _load_chain_audit() -> dict[str, Any] | None:
    path = SYSTEM_ROOT / "CHAIN_AUDIT.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def system_status() -> dict[str, Any]:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(RUNTIME_ROOT)
    weights = MODELS_ROOT / "yolov8x.pt"
    gpu = {"available": False, "name": None, "cuda": None, "device_count": 0}
    try:
        import torch
        gpu["available"] = bool(torch.cuda.is_available())
        gpu["cuda"] = getattr(torch.version, "cuda", None)
        gpu["device_count"] = int(torch.cuda.device_count()) if gpu["available"] else 0
        if gpu["available"]:
            gpu["name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    dependency_names = [
        "fastapi", "uvicorn", "multipart", "cv2", "numpy", "scipy", "yaml",
        "torch", "torchvision", "ultralytics", "easyocr", "skimage", "shapely", "openpyxl",
    ]
    dependencies = {name: importlib.util.find_spec(name) is not None for name in dependency_names}
    inference_required = [
        "cv2", "numpy", "scipy", "yaml", "torch", "torchvision", "ultralytics",
        "easyocr", "skimage", "shapely", "openpyxl",
    ]
    inference_deps_ready = all(dependencies.get(name, False) for name in inference_required)
    ffmpeg = _portable_ffmpeg()
    ffprobe = shutil.which("ffprobe")

    engine_checks = {
        "tracking": (SYSTEM_ROOT / "engine" / "tracking" / "run_pipeline.py").is_file(),
        "onboarding": (SYSTEM_ROOT / "engine" / "tracking" / "onboard" / "video_health.py").is_file(),
        "metric": (SYSTEM_ROOT / "engine" / "football_metric_running" / "src" / "running_metrics_v1").is_dir(),
        "dynamic_calibration": (SYSTEM_ROOT / "engine" / "football_metric_running" / "src" / "running_metrics_v1" / "build_multi_anchor_dynamic_calibration.py").is_file(),
        "match_analysis": (SYSTEM_ROOT / "engine" / "match_analysis" / "run_integrated_analysis.py").is_file(),
        "jersey_ocr": (SYSTEM_ROOT / "engine" / "match_analysis" / "run_jersey_ocr.py").is_file(),
        "player_cards": (SYSTEM_ROOT / "engine" / "match_analysis" / "generate_player_card.py").is_file(),
        "identity_audit": (SYSTEM_ROOT / "engine" / "identity_audit" / "mode_split" / "audit_mot.py").is_file(),
    }
    chain = _load_chain_audit()
    return {
        "version": SYSTEM_VERSION,
        "platform": {
            "system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "windows": platform.system().lower() == "windows",
        },
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "gpu": gpu,
        "model": {
            "ready": weights.is_file(), "path": str(weights),
            "size_bytes": weights.stat().st_size if weights.is_file() else 0,
        },
        "ffmpeg": {"ready": ffmpeg is not None, "path": ffmpeg, "source": "system" if shutil.which("ffmpeg") else ("imageio-ffmpeg" if ffmpeg else None)},
        "ffprobe": {"ready": ffprobe is not None, "path": ffprobe, "optional": True},
        "disk": {"runtime_root": str(RUNTIME_ROOT), "free_bytes": disk.free, "total_bytes": disk.total},
        "dependencies": dependencies,
        "readiness": {
            "web": bool(dependencies.get("fastapi") and dependencies.get("uvicorn") and dependencies.get("multipart")),
            "inference_dependencies": inference_deps_ready,
            # Core pipeline uses OpenCV for video IO; ffprobe is optional and must not block analysis.
            "inference": bool(inference_deps_ready and weights.is_file() and all(engine_checks.values())),
        },
        "engine": engine_checks,
        "chain_audit": {
            "available": bool(chain),
            "status": (chain or {}).get("status"),
            "critical_files_matched": (chain or {}).get("summary", {}).get("critical_files_matched"),
            "critical_files_total": (chain or {}).get("summary", {}).get("critical_files_total"),
        },
    }
