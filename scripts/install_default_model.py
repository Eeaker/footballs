from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TARGET = MODELS / "yolov8x.pt"


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    if TARGET.is_file() and TARGET.stat().st_size > 1_000_000:
        print(f"[OK] 默认模型已存在: {TARGET} ({TARGET.stat().st_size / 1024**2:.1f} MB)")
        return 0
    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"[ERROR] ultralytics 未安装: {exc}")
        print("请先运行 INSTALL_WINDOWS.bat 安装完整 AI 环境。")
        return 2

    old_cwd = Path.cwd()
    try:
        os.chdir(MODELS)
        print("正在通过 Ultralytics 下载默认 yolov8x.pt；首次下载需要联网，请稍候……")
        YOLO("yolov8x.pt")
    except Exception as exc:
        print(f"[ERROR] 模型下载失败: {exc}")
        print("也可以在系统状态页手动上传 yolov8x.pt。")
        return 3
    finally:
        os.chdir(old_cwd)

    # Ultralytics normally writes the requested asset into current working dir.
    candidate = MODELS / "yolov8x.pt"
    if not candidate.is_file():
        # Be defensive if a package version placed it in the caller cwd/cache.
        alt = old_cwd / "yolov8x.pt"
        if alt.is_file():
            shutil.move(str(alt), str(candidate))
    if not candidate.is_file() or candidate.stat().st_size <= 1_000_000:
        print("[ERROR] 下载流程结束但 models/yolov8x.pt 未找到。请改用网页上传模型。")
        return 4
    print(f"[OK] 模型已安装: {candidate} ({candidate.stat().st_size / 1024**2:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
