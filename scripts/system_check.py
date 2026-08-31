from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
checks: list[tuple[str, bool, str, bool]] = []


def add(name: str, ok: bool, detail: str, required: bool = True) -> None:
    checks.append((name, bool(ok), detail, required))


def portable_ffmpeg() -> str | None:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        return path if path and Path(path).is_file() else None
    except Exception:
        return None


# Windows production target is intentionally narrower than upstream library support.
py_supported = (3, 11) <= sys.version_info[:2] <= (3, 12)
add("Python 3.11/3.12", py_supported, f"{sys.version.split()[0]} · {platform.system()}")
for module in [
    "fastapi", "uvicorn", "multipart", "cv2", "numpy", "scipy", "yaml",
    "torch", "torchvision", "ultralytics", "easyocr", "skimage", "shapely", "openpyxl",
]:
    ok = importlib.util.find_spec(module) is not None
    add(f"Python: {module}", ok, "installed" if ok else "missing")

ffmpeg = portable_ffmpeg()
add("FFmpeg", ffmpeg is not None, ffmpeg or "not installed; core OpenCV pipeline can still run", required=False)
ffprobe = shutil.which("ffprobe")
add("ffprobe", ffprobe is not None, ffprobe or "optional", required=False)

required_paths = [
    ROOT / "app" / "app.py",
    ROOT / "app" / "services" / "pipeline.py",
    ROOT / "engine" / "tracking" / "run_pipeline.py",
    ROOT / "engine" / "tracking" / "onboard" / "video_health.py",
    ROOT / "engine" / "tracking" / "config" / "botsort_buffer.yaml",
    ROOT / "engine" / "football_metric_running" / "src" / "running_metrics_v1" / "build_multi_anchor_dynamic_calibration.py",
    ROOT / "engine" / "match_analysis" / "run_integrated_analysis.py",
    ROOT / "engine" / "match_analysis" / "run_jersey_ocr.py",
    ROOT / "engine" / "match_analysis" / "generate_player_card.py",
    ROOT / "engine" / "match_analysis" / "third_party" / "football-player-identification" / "ft" / "features" / "jersey_ocr.py",
    ROOT / "engine" / "identity_audit" / "mode_split" / "audit_mot.py",
    ROOT / "CHAIN_AUDIT.json",
]
for path in required_paths:
    add(str(path.relative_to(ROOT)), path.exists(), "ready" if path.exists() else "missing")

chain_ok = False
try:
    chain = json.loads((ROOT / "CHAIN_AUDIT.json").read_text(encoding="utf-8"))
    chain_ok = chain.get("status") == "passed" and chain.get("summary", {}).get("critical_files_matched") == chain.get("summary", {}).get("critical_files_total")
    add("原始算法链路 SHA256 审计", chain_ok, f"{chain.get('summary', {}).get('critical_files_matched')}/{chain.get('summary', {}).get('critical_files_total')}")
except Exception as exc:
    add("原始算法链路 SHA256 审计", False, str(exc))

weights = ROOT / "models" / "yolov8x.pt"
add("正式分析权重", weights.is_file(), str(weights) if weights.is_file() else "未放置：可在网页系统状态上传 yolov8x.pt", required=False)

gpu_ok = False
try:
    import torch
    gpu_ok = bool(torch.cuda.is_available())
    detail = torch.cuda.get_device_name(0) if gpu_ok else "CUDA unavailable; CPU mode remains available"
    add("CUDA GPU", gpu_ok, detail, required=False)
except Exception as exc:
    add("CUDA GPU", False, str(exc), required=False)

print("\nFootball Insight V2.3.3 Windows/Production 系统检查\n" + "=" * 74)
for name, ok, detail, required in checks:
    state = "PASS" if ok else ("WARN" if not required else "FAIL")
    print(f"{state:<5} {name:<48} {detail}")
print("=" * 74)
required_ok = all(ok for _, ok, _, required in checks if required)
print("完整程序链路：", "READY" if required_ok and chain_ok else "NOT READY")
print("正式新视频分析：", "READY" if required_ok and weights.is_file() else "WAITING FOR MODEL" if required_ok else "NOT READY")
print("推荐计算设备：", "NVIDIA GPU" if gpu_ok else "CPU（可运行但长视频很慢）")
sys.exit(0 if required_ok else 1)
