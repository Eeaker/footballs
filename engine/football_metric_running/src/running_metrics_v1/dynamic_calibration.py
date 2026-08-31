"""Per-frame homography preparation for moving-camera footage.

This is an independent implementation of the per-frame court-registration
architecture reviewed in abdullahtarek/basketball_analysis. No source code
from that repository is copied; third-party use is described in
``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

from .homography import point_reprojection_errors_m, solve_image_to_world


@dataclass(frozen=True)
class FrameCalibrationResult:
    proc_idx: int
    accepted: bool
    homography: list[list[float]] | None
    point_count: int
    reprojection_rmse_m: float | None
    reprojection_max_m: float | None
    reason: str | None


def solve_frame_calibrations(
    observations: Iterable[dict],
    max_reprojection_error_m: float = 0.5,
) -> tuple[dict[int, np.ndarray], list[FrameCalibrationResult]]:
    """Solve and quality-check one image-to-meter homography per frame.

    Each observation needs ``proc_idx``, ``image_points`` and
    ``world_points_m``. Rejected frames deliberately have no fallback H, so
    downstream distance code splits the trajectory instead of bridging an
    uncertain camera pose.
    """
    if max_reprojection_error_m < 0:
        raise ValueError("max_reprojection_error_m must be non-negative")

    matrices: dict[int, np.ndarray] = {}
    results: list[FrameCalibrationResult] = []
    seen: set[int] = set()
    for item in observations:
        proc_idx = int(item["proc_idx"])
        if proc_idx < 0:
            raise ValueError("proc_idx must be non-negative")
        if proc_idx in seen:
            raise ValueError(f"duplicate frame calibration: proc_idx={proc_idx}")
        seen.add(proc_idx)
        image = item.get("image_points", [])
        world = item.get("world_points_m", [])
        point_count = len(image)
        matrix = None
        rmse = maximum = None
        reason = None
        try:
            if point_count != len(world):
                raise ValueError("image/world point counts differ")
            matrix = solve_image_to_world(image, world)
            errors = point_reprojection_errors_m(matrix, image, world)
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            maximum = float(np.max(errors))
            if maximum > max_reprojection_error_m:
                reason = "reprojection_error"
                matrix = None
        except (ValueError, cv2.error) as exc:
            reason = str(exc)
            matrix = None

        accepted = matrix is not None
        if accepted:
            matrices[proc_idx] = matrix
        results.append(
            FrameCalibrationResult(
                proc_idx=proc_idx,
                accepted=accepted,
                homography=matrix.tolist() if accepted else None,
                point_count=point_count,
                reprojection_rmse_m=rmse,
                reprojection_max_m=maximum,
                reason=reason,
            )
        )
    return matrices, sorted(results, key=lambda result: result.proc_idx)


def homographies_from_calibration(calibration: Mapping) -> dict[int, np.ndarray]:
    """Load accepted per-frame matrices from a dynamic calibration document."""
    if calibration.get("camera_model") != "dynamic_per_frame_homography":
        raise ValueError("not a dynamic_per_frame_homography calibration")
    matrices: dict[int, np.ndarray] = {}
    for frame in calibration.get("frames", []):
        if not frame.get("accepted", True) or frame.get("H_image_to_pitch_m") is None:
            continue
        proc_idx = int(frame["proc_idx"])
        matrix = np.asarray(frame["H_image_to_pitch_m"], dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid homography at proc_idx={proc_idx}")
        matrices[proc_idx] = matrix
    return matrices
