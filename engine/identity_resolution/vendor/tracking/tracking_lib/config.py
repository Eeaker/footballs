from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PIPELINE_MAP = {
    "video.path": "video", "detector.weights": "weights", "detector.confidence": "conf",
    "detector.imgsz": "imgsz", "tracker.config_file": "tracker_config",
    "tracker.vid_stride": "vid_stride", "scene.expected_on_field_players": "expected_on_field_players",
    "identity.max_ids": "max_ids", "identity.team_clusters": "team_clusters",
    "highlights.event_count": "n_clips", "highlights.seconds_before_event": "pre_sec",
    "highlights.seconds_after_event": "post_sec",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 onboard 生成的场地 YAML 配置。"""
    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("配置文件顶层必须为映射")
    data["_path"] = str(source.resolve())
    return data


def get_value(data: dict, dotted: str, default: Any = None) -> Any:
    """按点分路径安全读取嵌套配置。"""
    value: Any = data
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def resolve_config_path(data: dict, dotted: str, default: str | Path | None = None) -> Path | None:
    """读取路径字段，并相对配置文件所在目录解析。"""
    value = get_value(data, dotted, default)
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute() and data.get("_path"):
        path = Path(data["_path"]).parent / path
    return path.resolve()


def argparse_defaults(data: dict) -> dict[str, Any]:
    """将公共场地配置转换成底层跟踪入口参数，供兼容性检查使用。"""
    result = {}
    for source, destination in PIPELINE_MAP.items():
        value = get_value(data, source)
        if value is not None:
            result[destination] = value
    return result
