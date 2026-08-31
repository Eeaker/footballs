"""Planar image-to-pitch calibration helpers.

The homogeneous point projection convention follows the MIT-licensed
TVCalib pixel-to-world example.  See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class SegmentValidation:
    name: str
    known_length_m: float
    measured_length_m: float
    absolute_error_m: float
    passed: bool


def _points(value: Iterable[Sequence[float]], name: str) -> np.ndarray:
    points = np.asarray(list(value), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return points


def solve_image_to_world(
    image_points: Iterable[Sequence[float]],
    world_points_m: Iterable[Sequence[float]],
) -> np.ndarray:
    """Estimate an image-pixel to pitch-meter homography from >=4 pairs."""
    image = _points(image_points, "image_points")
    world = _points(world_points_m, "world_points_m")
    if image.shape != world.shape:
        raise ValueError("image_points and world_points_m must have equal shape")
    if len(image) < 4:
        raise ValueError("at least four point pairs are required")

    image_hull_area = abs(cv2.contourArea(cv2.convexHull(image.astype(np.float32))))
    world_hull_area = abs(cv2.contourArea(cv2.convexHull(world.astype(np.float32))))
    if image_hull_area < 1.0 or world_hull_area < 1e-6:
        raise ValueError("calibration points are degenerate or nearly collinear")

    if len(image) == 4:
        matrix = cv2.getPerspectiveTransform(
            image.astype(np.float32), world.astype(np.float32)
        ).astype(np.float64)
    else:
        matrix, mask = cv2.findHomography(image, world, method=0)
        if matrix is None or mask is None:
            raise ValueError("OpenCV could not estimate a homography")

    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-15:
        raise ValueError("estimated homography is singular")
    matrix /= matrix[2, 2]
    return matrix


def project_points(points_xy: Iterable[Sequence[float]], matrix: np.ndarray) -> np.ndarray:
    """Project N two-dimensional points with a 3x3 homography."""
    points = _points(points_xy, "points_xy")
    h = np.asarray(matrix, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = (h @ homogeneous.T).T
    denominator = projected[:, 2]
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError("a projected point lies at infinity")
    return projected[:, :2] / denominator[:, None]


def validate_segments(
    matrix: np.ndarray,
    segments: Iterable[dict],
    tolerance_m: float = 0.5,
) -> list[SegmentValidation]:
    """Validate H using known line segments not used to solve it."""
    if tolerance_m < 0:
        raise ValueError("tolerance_m must be non-negative")
    results: list[SegmentValidation] = []
    for index, segment in enumerate(segments, start=1):
        points = segment.get("image_points")
        if points is None or len(points) != 2:
            raise ValueError(f"validation segment {index} needs two image_points")
        known = float(segment["known_length_m"])
        if known <= 0:
            raise ValueError("known_length_m must be positive")
        world = project_points(points, matrix)
        measured = float(np.linalg.norm(world[1] - world[0]))
        error = abs(measured - known)
        results.append(
            SegmentValidation(
                name=str(segment.get("name", f"segment_{index}")),
                known_length_m=known,
                measured_length_m=measured,
                absolute_error_m=error,
                passed=error <= tolerance_m,
            )
        )
    return results


def point_reprojection_errors_m(
    matrix: np.ndarray,
    image_points: Iterable[Sequence[float]],
    world_points_m: Iterable[Sequence[float]],
) -> np.ndarray:
    image = _points(image_points, "image_points")
    world = _points(world_points_m, "world_points_m")
    if image.shape != world.shape:
        raise ValueError("point arrays must have equal shape")
    return np.linalg.norm(project_points(image, matrix) - world, axis=1)
