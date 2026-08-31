from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import yaml

from .tracking_adapter import project_footpoint as _tracking_project_footpoint


def project_point(homography: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    result = _tracking_project_footpoint(homography, x, y, 0.0, 0.0)
    return result if np.isfinite(result).all() else None


@dataclass(frozen=True)
class CalibrationInfo:
    source: Path
    validated: bool
    mode: str
    video_name: str | None
    width: int | None
    height: int | None
    vid_stride: int


class HomographyProvider:
    """Read tracking YAML keyframes or the per-frame running-metrics JSON format."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if self.path.suffix.lower() == ".json":
            data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self._raw = data
        nested_calibration = data.get("calibration", {}) if isinstance(data.get("calibration"), dict) else {}
        bounds = data.get("field_bounds_m", nested_calibration.get("field_bounds_m", {}))
        self.field_bounds_m = bounds if isinstance(bounds, dict) else {}
        self._per_frame: dict[int, np.ndarray] = {}
        self._rejected_frames: set[int] = set()
        self._keyframes: list[tuple[int, np.ndarray]] = []
        if "frames" in data and data.get("camera_model"):
            for row in data.get("frames", []):
                matrix = row.get("H_image_to_pitch_m")
                if row.get("accepted", True) and matrix is not None:
                    self._per_frame[int(row["proc_idx"])] = np.asarray(matrix, dtype=np.float64)
                else:
                    self._rejected_frames.add(int(row["proc_idx"]))
            meta = data.get("video_metadata", {})
            validation = data.get("validation", {})
            self.info = CalibrationInfo(
                self.path, bool(validation.get("passed")), str(data.get("camera_model")),
                _portable_basename(data.get("video")),
                _optional_int(meta.get("frame_width")), _optional_int(meta.get("frame_height")),
                int(data.get("vid_stride", 1)),
            )
        else:
            calibration = data.get("calibration", data)
            for row in calibration.get("keyframes", []):
                matrix = row.get("homography")
                if matrix is not None:
                    self._keyframes.append((int(row["frame_index"]), np.asarray(matrix, dtype=np.float64)))
            self._keyframes.sort(key=lambda item: item[0])
            video = data.get("video", {}) if isinstance(data.get("video"), dict) else {}
            self.info = CalibrationInfo(
                self.path, bool(calibration.get("validated")), str(calibration.get("mode", "keyframes")),
                _portable_basename(video.get("path")),
                _optional_int(video.get("width")), _optional_int(video.get("height")),
                int(data.get("tracker", {}).get("vid_stride", 1)) if isinstance(data.get("tracker"), dict) else 1,
            )
        if not self._per_frame and not self._keyframes:
            raise ValueError(f"标定文件不含可用单应矩阵: {self.path}")
        if not self.info.validated:
            raise ValueError(f"标定未通过独立验证，拒绝输出米制球权: {self.path}")

    def in_field(self, point: tuple[float, float]) -> bool:
        """Apply the field bounds already carried by the calibration artifact."""
        if not self.field_bounds_m:
            return True
        margin = float(self.field_bounds_m.get("margin_m", 0.0))
        x, y = point
        return (
            float(self.field_bounds_m["x_min"]) - margin <= x <= float(self.field_bounds_m["x_max"]) + margin
            and float(self.field_bounds_m["y_min"]) - margin <= y <= float(self.field_bounds_m["y_max"]) + margin
        )

    def at_processed_frame(self, frame_proc: int, vid_stride: int) -> np.ndarray | None:
        if self._per_frame:
            # A dense dynamic calibration explicitly marks registration gaps.
            # Never replace those invalid frames with a distant nearest matrix.
            if frame_proc in self._rejected_frames:
                return None
            if frame_proc in self._per_frame:
                return self._per_frame[frame_proc]
            keys = sorted(self._per_frame)
            nearest = min(keys, key=lambda key: abs(key - frame_proc))
            return self._per_frame[nearest]
        raw_frame = frame_proc * max(1, vid_stride)
        assert self._keyframes
        if raw_frame <= self._keyframes[0][0]:
            return self._keyframes[0][1]
        if raw_frame >= self._keyframes[-1][0]:
            return self._keyframes[-1][1]
        for (left_f, left_h), (right_f, right_h) in zip(self._keyframes, self._keyframes[1:]):
            if left_f <= raw_frame <= right_f:
                ratio = (raw_frame - left_f) / max(right_f - left_f, 1)
                left = left_h / left_h[2, 2]
                right = right_h / right_h[2, 2]
                result = left * (1.0 - ratio) + right * ratio
                return result / result[2, 2]
        return None


def validate_calibration_compatibility(
    provider: HomographyProvider,
    *,
    expected_video_name: str | None,
    width: int | None,
    height: int | None,
    vid_stride: int,
) -> None:
    info = provider.info
    errors = []
    if expected_video_name and info.video_name and expected_video_name.lower() != info.video_name.lower():
        errors.append(f"视频名 {expected_video_name!r} != 标定视频 {info.video_name!r}")
    if width and info.width and width != info.width:
        errors.append(f"宽度 {width} != 标定宽度 {info.width}")
    if height and info.height and height != info.height:
        errors.append(f"高度 {height} != 标定高度 {info.height}")
    if info.vid_stride != vid_stride:
        errors.append(f"vid_stride {vid_stride} != 标定 vid_stride {info.vid_stride}")
    if errors:
        raise ValueError("标定与追踪结果不匹配；禁止套用其他视频矩阵: " + "; ".join(errors))


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _portable_basename(value: Any) -> str | None:
    """Extract a video basename even when an artifact crosses Windows/Linux."""
    text = str(value or "").strip()
    if not text:
        return None
    if "\\" in text:
        return PureWindowsPath(text).name or None
    return Path(text).name or None
