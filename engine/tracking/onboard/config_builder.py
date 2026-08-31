from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

import yaml

from .models import CalibrationResult, MotionHealth, TeamColorResult, TrialMetrics


def safe_venue_name(name: str) -> str:
    """生成可用作跨平台目录名的场地标识。"""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name.strip())
    return cleaned.strip(" .") or "new_venue"


def load_tracker_yaml(path: str | Path) -> dict:
    """读取 BoT-SORT YAML。"""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def candidate_tracker_config(baseline: dict, health: MotionHealth) -> dict:
    """根据相机类型生成保守候选参数，最终是否采用由 Stage D 决定。"""
    result = deepcopy(baseline)
    result["gmc_method"] = "sparseOptFlow"
    if health.motion_type == "fixed":
        result["track_buffer"] = max(48, int(round(health.metadata.fps * 2.0)))
        result["match_thresh"] = .80
    elif health.motion_type == "pan_rotate":
        result["track_buffer"] = max(72, int(round(health.metadata.fps * 3.0)))
        result["match_thresh"] = .82
    else:
        result["track_buffer"] = max(96, int(round(health.metadata.fps * 4.0)))
        result["match_thresh"] = .85
    return result


def build_pipeline_config(venue: str, video_path: str | Path, health: MotionHealth,
                          colors: TeamColorResult | None, calibration: CalibrationResult,
                          tracker_file: str | Path, trial: TrialMetrics | None,
                          expected_players: int = 16, referee_present: bool = False,
                          weights: str | Path = "models/yolov8x.pt",
                          manual: dict | None = None,
                          field_geometry: dict | None = None) -> dict:
    """生成与 tracking 主流程一一映射的场地配置。"""
    manual = manual or {}
    return {
        "schema_version": 1,
        "venue": safe_venue_name(venue),
        "video": {"path": str(Path(video_path).resolve()), "fps": health.metadata.fps,
                  "width": health.metadata.width, "height": health.metadata.height,
                  "duration_seconds": round(health.metadata.duration_seconds, 3)},
        "scene": {"camera_motion": health.motion_type, "expected_on_field_players": expected_players,
                  "referee_present": referee_present},
        "detector": {"weights": str(Path(weights).resolve()),
                     "confidence": float(manual.get("confidence", .25)),
                     "imgsz": int(manual.get("imgsz", 1280)),
                     "person_class": 0},
        "tracker": {"config_file": str(Path(tracker_file).resolve()),
                    "vid_stride": int(manual.get("vid_stride", 1)),
                    "selected_by_trial": trial.name if trial else "not_run"},
        "field_filter": {"enabled": bool(manual.get("field_filter_enabled", True)),
                         "min_turf_score": float(manual.get("min_turf_score", .15)),
                         "min_track_turf_ratio": float(manual.get("min_track_turf_ratio", .25)),
                         "min_foot_y_ratio": float(manual.get("min_foot_y_ratio", .32)),
                         "min_geometry_ratio": float(manual.get("min_geometry_ratio", .60)),
                         "geometry": field_geometry or {"enabled": False, "mode": "disabled"}},
        "identity": {"max_ids": expected_players, "keep_all_clusters": True,
                     "min_track_frames": int(manual.get("min_track_frames", 10)),
                     "min_presence_ratio": float(manual.get("min_presence_ratio", .005)),
                     "team_clusters": int(manual.get("team_clusters", colors.selected_k if colors else 2)),
                     "team_names": [cluster.name for cluster in colors.clusters] if colors else [],
                     "team_prototypes": [
                         {"cluster_id": cluster.cluster_id, "name": cluster.name,
                          "suggested_color": cluster.suggested_color,
                          "feature_center": cluster.feature_center}
                         for cluster in colors.clusters
                     ] if colors else []},
        "calibration": {**calibration.to_dict(),
                        "dynamic_update": "normalized_homography_interpolation"
                        if calibration.enabled and calibration.mode not in {"static", "disabled"} else "none"},
        "metric_motion": {"enabled": calibration.enabled and calibration.validated,
                          "max_playback_speed": 8, "footpoint": "bottom_center",
                          "distance_smoothing_window": 5},
        "events": {"percentile": float(manual.get("event_percentile", 92.0)),
                   "minimum_gap_seconds": float(manual.get("event_min_gap", 2.0)),
                   "edge_margin_seconds": float(manual.get("edge_margin", 20.0))},
        "highlights": {"event_count": int(manual.get("event_count", 5)),
                       "seconds_before_event": float(manual.get("pre_sec", 15.0)),
                       "seconds_after_event": float(manual.get("post_sec", 15.0))},
        "review": {"evaluation_policy": "eight_dimension_candidate_labels_plus_human_review",
                   "multimodal_direct_scoring": False},
    }


def write_yaml(path: str | Path, data: dict) -> Path:
    """以可读、保持中文的格式写 YAML。"""
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target
