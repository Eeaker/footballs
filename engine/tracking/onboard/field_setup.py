from __future__ import annotations

import json
from pathlib import Path

from .ui import annotate_video_keyframes


def load_field_geometry(path: str | Path) -> dict:
    """加载已确认的场地几何 JSON。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "mode" not in data:
        raise ValueError("场地几何 JSON 格式非法")
    return data


def capture_pitch_polygons(video_path: str | Path, total_frames: int, dynamic: bool,
                           margin_px: float = 12.0) -> dict:
    """人工勾画比赛区域；动态镜头在多个关键帧保存多边形。"""
    suggested = ([max(0, int(total_frames * ratio)) for ratio in (.05, .25, .50, .75, .95)]
                 if dynamic else [max(0, int(total_frames * .05))])
    keyframes = annotate_video_keyframes(
        video_path, suggested,
        title=("Dynamic visible-field polygon calibration" if dynamic
               else "Visible-field polygon calibration"),
        minimum=4, maximum=8, close_shape=True,
        minimum_keyframes=2 if dynamic else 1,
    )
    if not keyframes:
        return {"enabled": False, "mode": "disabled", "reason": "user_cancelled"}
    return {"enabled": True, "mode": "polygon_keyframes", "margin": float(margin_px),
            "point_order": "clockwise_or_counterclockwise_consistent",
            "vertices_per_keyframe": "4_to_8_variable",
            "interpolation": "linear_if_same_vertex_count_else_nearest_keyframe",
            "keyframes": keyframes}
