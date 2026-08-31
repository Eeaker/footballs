from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from app.config import ENGINE_ROOT


def inspect_uploaded_video(video_path: str | Path, sample_pairs: int = 32) -> dict[str, Any]:
    """Run the original repository's Stage-A camera-motion health inspection.

    This is advisory only. Football Insight's formal product policy for the
    user's rotating fixed-position cameras remains dynamic multi-anchor
    calibration even if a short clip happens to look static.
    """
    tracking_root = ENGINE_ROOT / "tracking"
    path_text = str(tracking_root)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    from onboard.video_health import inspect_video_health  # type: ignore

    health = inspect_video_health(video_path, sample_pairs=max(8, min(96, int(sample_pairs))))
    data = health.to_dict()
    data["product_calibration_policy"] = "dynamic_multi_anchor"
    data["product_note"] = "固定拍摄点但存在旋转视角，正式米制分析统一使用多锚点逐帧动态标定。"
    return data
