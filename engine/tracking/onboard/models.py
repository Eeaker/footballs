from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class Serializable:
    """为阶段结果提供稳定的 JSON/YAML 序列化接口。"""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VideoMetadata(Serializable):
    path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float


@dataclass
class MotionHealth(Serializable):
    metadata: VideoMetadata
    motion_type: str
    dynamic_h_usable: bool
    calibration_mode: str
    median_translation_px: float
    median_rotation_deg: float
    median_inlier_ratio: float
    median_residual_px: float
    valid_pairs: int
    recommendation: str
    p75_translation_px: float = 0.0
    p90_translation_px: float = 0.0
    moving_pair_ratio: float = 0.0


@dataclass
class TeamCluster(Serializable):
    cluster_id: int
    count: int
    median_hsv: list[float]
    representative_bgr: list[int]
    suggested_color: str = "unknown"
    feature_center: list[float] = field(default_factory=list)
    quality_score: float = 0.0
    name: str = ""


@dataclass
class TeamColorResult(Serializable):
    sample_frames: int
    person_crops: int
    silhouette_k2: float
    silhouette_k3: float
    recommended_k: int
    selected_k: int
    clusters: list[TeamCluster] = field(default_factory=list)
    board_path: str = ""


@dataclass
class CalibrationKeyframe(Serializable):
    frame_index: int
    image_points: list[list[float]]
    world_points_m: list[list[float]]
    homography: list[list[float]]
    fit_rmse_m: float
    validation_error_m: float


@dataclass
class CalibrationResult(Serializable):
    enabled: bool
    mode: str
    coordinate_system: str = "origin_bottom_left_x_length_y_width_meters"
    keyframes: list[CalibrationKeyframe] = field(default_factory=list)
    validation_threshold_m: float = 0.5
    validated: bool = False


@dataclass
class TrialMetrics(Serializable):
    name: str
    tracker_config: str
    processed_frames: int
    processed_seconds: float
    mean_boxes_per_frame: float
    local_id_total: int
    new_ids_per_minute: float
    median_track_length_frames: float
    short_tracks_lt_10: int
