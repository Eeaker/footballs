from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .homography import project_footpoint


@dataclass(frozen=True)
class MetricFieldGateResult:
    detections: list[tuple]
    report: dict
    track_positions_m: dict[int, list[tuple[int, float, float]]]


def _inside(x: float, y: float, bounds: dict, margin: float) -> bool:
    return (
        float(bounds["x_min"]) - margin <= x <= float(bounds["x_max"]) + margin
        and float(bounds["y_min"]) - margin <= y <= float(bounds["y_max"]) + margin
    )


def filter_detections_by_dynamic_pitch(
    detections: list[tuple], calibration_path: str | Path, *,
    association_margin_m: float = 0.0, hard_outside_margin_m: float = 0.0,
    minimum_track_inside_ratio: float = .5, minimum_track_inside_frames: int = 10,
) -> MetricFieldGateResult:
    """Keep only tracklets with sustained evidence inside the dynamic pitch.

    The original local MOT should be persisted before calling this function.
    Missing or rejected calibration is a hard error: silently applying a static
    or partial gate to a rotating camera would be less safe than disabling it.
    """
    source = Path(calibration_path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not bool((data.get("validation") or {}).get("passed")):
        raise ValueError("metric field gate requires a passed dynamic calibration")
    bounds = data.get("field_bounds_m")
    if not isinstance(bounds, dict):
        raise ValueError("dynamic calibration has no field_bounds_m")
    matrices = {
        int(row["proc_idx"]): np.asarray(row["H_image_to_pitch_m"], np.float64)
        for row in data.get("frames", [])
        if row.get("accepted") and row.get("H_image_to_pitch_m") is not None
    }
    if not matrices:
        raise ValueError("dynamic calibration has no accepted frame homographies")

    by_track = defaultdict(list)
    missing_calibration = 0
    hard_outside = 0
    projected_rows = []
    for index, row in enumerate(detections):
        frame, track_id, x, y, width, height, *_ = row
        matrix = matrices.get(int(frame))
        if matrix is None:
            missing_calibration += 1
            projected_rows.append((row, False, False, None, None))
            continue
        pitch_x, pitch_y = project_footpoint(matrix, x, y, width, height)
        finite = bool(np.isfinite(pitch_x) and np.isfinite(pitch_y))
        inside_association = finite and _inside(
            pitch_x, pitch_y, bounds, association_margin_m
        )
        inside_hard = finite and _inside(
            pitch_x, pitch_y, bounds, hard_outside_margin_m
        )
        if not inside_hard:
            hard_outside += 1
        projected_rows.append((row, inside_association, inside_hard,
                               float(pitch_x) if finite else None, float(pitch_y) if finite else None))
        by_track[int(track_id)].append((index, inside_association, inside_hard))

    if missing_calibration:
        raise ValueError(
            f"dynamic calibration missing for {missing_calibration} tracking detections"
        )

    kept_tracks = set()
    track_rows = []
    for track_id, observations in sorted(by_track.items()):
        inside = sum(item[1] for item in observations)
        ratio = inside / max(len(observations), 1)
        accepted = inside >= minimum_track_inside_frames and ratio >= minimum_track_inside_ratio
        if accepted:
            kept_tracks.add(track_id)
        track_rows.append({
            "local_track_id": track_id, "detections": len(observations),
            "association_inside_frames": inside,
            "association_inside_ratio": round(ratio, 6), "accepted": accepted,
        })

    filtered = [
        row for row, inside_association, _, _, _ in projected_rows
        if int(row[1]) in kept_tracks and inside_association
    ]
    track_positions = defaultdict(list)
    for row, inside_association, _, pitch_x, pitch_y in projected_rows:
        if int(row[1]) in kept_tracks and inside_association:
            track_positions[int(row[1])].append((int(row[0]), float(pitch_x), float(pitch_y)))
    return MetricFieldGateResult(filtered, {
        "enabled": True,
        "method": "dynamic_homography_bbox_footpoint_before_global_association",
        "calibration": str(source.resolve()),
        "field_bounds_m": bounds,
        "association_margin_m": association_margin_m,
        "hard_outside_margin_m": hard_outside_margin_m,
        "minimum_track_inside_ratio": minimum_track_inside_ratio,
        "minimum_track_inside_frames": minimum_track_inside_frames,
        "input_detections": len(detections), "output_detections": len(filtered),
        "hard_outside_detections": hard_outside,
        "input_tracklets": len(by_track), "kept_tracklets": len(kept_tracks),
        "rejected_tracklets": len(by_track) - len(kept_tracks),
        "tracks": track_rows,
    }, dict(track_positions))
