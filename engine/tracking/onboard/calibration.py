from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import CalibrationKeyframe, CalibrationResult


def compute_homography(image_points: list[list[float]], world_points: list[list[float]]) -> tuple[np.ndarray, float]:
    """由 4~8 对图像/米制点稳健求 H，并返回拟合 RMSE。"""
    if not 4 <= len(image_points) <= 8 or len(image_points) != len(world_points):
        raise ValueError("标定必须提供数量一致的 4~8 对点")
    src, dst = np.asarray(image_points, np.float64), np.asarray(world_points, np.float64)
    homography, _ = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    if homography is None:
        raise ValueError("参照点退化，无法求解 Homography")
    projected = cv2.perspectiveTransform(src[None].astype(np.float32), homography)[0]
    rmse = float(np.sqrt(np.mean(np.sum((projected - dst) ** 2, axis=1))))
    return homography, rmse


def project_points(homography: np.ndarray, points: list[list[float]]) -> np.ndarray:
    """将图像点映射到以米为单位的场地平面。"""
    return cv2.perspectiveTransform(np.asarray(points, np.float32)[None], homography)[0]


def validate_segments(homography: np.ndarray, segments: list[dict]) -> float:
    """用未参与拟合的独立线段验证米制长度，返回最大绝对误差。"""
    errors = []
    for segment in segments:
        endpoints = project_points(homography, [segment["p1"], segment["p2"]])
        predicted = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        errors.append(abs(predicted - float(segment["length_m"])))
    return max(errors, default=float("inf"))


def create_keyframe(frame_index: int, image_points: list[list[float]], world_points: list[list[float]],
                    validation_segments: list[dict]) -> CalibrationKeyframe:
    """生成一个经过独立线段验证的标定关键帧。"""
    homography, rmse = compute_homography(image_points, world_points)
    error = validate_segments(homography, validation_segments)
    return CalibrationKeyframe(frame_index, image_points, world_points, homography.tolist(), rmse, error)


def load_imported_calibration(path: str | Path) -> CalibrationResult:
    """加载可复用的标定 JSON，便于无界面部署。"""
    import json
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    keyframes = [CalibrationKeyframe(**item) for item in data.get("keyframes", [])]
    return CalibrationResult(bool(data.get("enabled", True)), data.get("mode", "static"),
        data.get("coordinate_system", "origin_bottom_left_x_length_y_width_meters"), keyframes,
        float(data.get("validation_threshold_m", .5)), bool(data.get("validated", False)))


