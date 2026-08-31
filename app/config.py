from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
SYSTEM_ROOT = APP_DIR.parent
ENGINE_ROOT = SYSTEM_ROOT / "engine"
DEMO_ROOT = SYSTEM_ROOT / "demo_data" / "reference_match"
MODELS_ROOT = Path(os.getenv("FOOTBALL_INSIGHT_MODELS", str(SYSTEM_ROOT / "models"))).resolve()
RUNTIME_ROOT = Path(os.getenv("FOOTBALL_INSIGHT_RUNTIME", str(SYSTEM_ROOT / "runtime"))).resolve()
PROJECTS_ROOT = RUNTIME_ROOT / "projects"
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"

PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
MODELS_ROOT.mkdir(parents=True, exist_ok=True)

SYSTEM_VERSION = "2.3.3"
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

DEFAULT_SETTINGS = {
    # Match / scene
    "expected_players": 10,
    "team_clusters": 3,
    # Compute
    "device": "0",
    "weights_path": str(MODELS_ROOT / "yolov8x.pt"),
    "confidence": 0.25,
    "imgsz": 1280,
    "vid_stride": 1,  # dynamic calibration requires raw-frame domain
    # Tracking / field filtering
    "min_track_frames": 10,
    "min_presence_ratio": 0.005,
    "min_turf_score": 0.15,
    "min_track_turf_ratio": 0.25,
    "min_foot_y_ratio": 0.55,
    # Event / highlight
    "event_percentile": 92.0,
    "event_min_gap": 2.0,
    "event_count": 20,
    "pre_sec": 3.0,
    "post_sec": 2.0,
    "min_pass_displacement_m": 0.5,
    # Team / OCR
    "team_samples_per_id": 12,
    "ocr_candidates_per_id": 36,
    # Identity quality audit (does not mutate upstream MOT)
    "identity_audit_enabled": True,
    "identity_audit_sample_stride": 30,
    # Metric pitch / calibration
    "field_length_m": 45.0,
    "field_width_m": 25.0,
    "calibration_tolerance_m": 0.5,
    "dynamic_sample_step": 5,
    "dynamic_max_gap": 30,
    "dynamic_min_coverage": 0.80,
    # Product output
    "focus_clip_limit": 12,
    "replay_max_frames": 15000,
}

# The boss-facing product progress is intentionally four steps.
PIPELINE_STEPS = [
    {"key": "tracking", "label": "追踪中", "hint": "识别人、球与连续轨迹"},
    {"key": "jersey", "label": "号码识别", "hint": "聚合多帧球衣号码证据"},
    {"key": "events", "label": "事件检测", "hint": "生成米制跑动、球权与传球结果"},
    {"key": "report", "label": "报告生成", "hint": "生成沙盘、高光、球员卡与报告"},
]

RESULT_SECTIONS = ["overview", "replay", "events", "highlights", "players", "quality", "report", "exports"]
