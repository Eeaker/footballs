from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .field_geometry import FieldGeometryProvider
from .team_features import aggregate_features


@dataclass(frozen=True)
class FieldFilterResult:
    detections: list[tuple]
    kept_track_ids: set[int]
    report: dict


def turf_support_score(frame: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """Estimate whether the bottom of a person box is supported by green turf.

    The patch deliberately extends around and slightly below the feet.  This is
    more tolerant of white pitch lines than sampling only the bottom-center
    pixel, while spectators standing on concrete/stands receive a low score.
    """
    if frame.size == 0:
        return 0.0
    x, y, w, h = map(float, box)
    frame_h, frame_w = frame.shape[:2]
    cx = x + 0.5 * w
    foot_y = y + h
    half_w = max(3, int(round(0.45 * w)))
    patch_h = max(4, int(round(0.18 * h)))
    x1 = max(0, int(round(cx)) - half_w)
    x2 = min(frame_w, int(round(cx)) + half_w + 1)
    y1 = max(0, int(round(foot_y - 0.06 * h)))
    y2 = min(frame_h, int(round(foot_y)) + patch_h + 1)
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    green = (hue >= 28) & (hue <= 100) & (sat >= 35) & (val >= 25)
    return float(green.mean())


def filter_tracklets_by_turf(
    video: str | Path,
    detections: Iterable[tuple],
    *,
    vid_stride: int = 1,
    min_detection_score: float = 0.15,
    min_track_ratio: float = 0.25,
    min_track_samples: int = 3,
    min_foot_y_ratio: float = 0.0,
    field_geometry: dict | None = None,
    min_geometry_ratio: float = 0.0,
) -> FieldFilterResult:
    """Remove tracklets that are consistently unsupported by pitch turf.

    Turf filtering is decided per tracklet, so a real player crossing a white
    line is retained.  The optional foot-point ROI is a per-observation hard
    geometry gate: it removes stands above the calibrated pitch horizon even
    when trees or nearby turf make their colour patch look green.
    """
    rows = list(detections)
    by_frame: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        by_frame[int(row[0])].append((row_index, row))

    scores: dict[int, list[float]] = defaultdict(list)
    geometry_votes: dict[int, list[bool]] = defaultdict(list)
    geometry_supported_rows: set[int] = set()
    geometry = FieldGeometryProvider(field_geometry)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"无法读取视频进行球场过滤: {video}")
    raw_idx = -1
    proc_idx = -1
    stride = max(1, int(vid_stride))
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        raw_idx += 1
        if (raw_idx + 1) % stride != 0:
            continue
        proc_idx += 1
        for row_index, row in by_frame.get(proc_idx, []):
            _, track_id, x, y, w, h, *_ = row
            scores[int(track_id)].append(turf_support_score(frame, (x, y, w, h)))
            foot = (float(x) + float(w) / 2.0, float(y) + float(h))
            roi_ok = foot[1] >= frame.shape[0] * min_foot_y_ratio
            field_ok = geometry.contains(raw_idx, foot) if geometry.enabled else True
            geometry_votes[int(track_id)].append(bool(roi_ok and field_ok))
            if roi_ok and field_ok:
                geometry_supported_rows.add(row_index)
    capture.release()

    kept: set[int] = set()
    per_track = []
    for track_id, values in scores.items():
        supported = sum(value >= min_detection_score for value in values)
        ratio = supported / max(len(values), 1)
        votes = geometry_votes.get(track_id, [])
        geometry_ratio = sum(votes) / max(len(votes), 1)
        accepted = (len(values) < min_track_samples or ratio >= min_track_ratio) and geometry_ratio >= min_geometry_ratio
        if accepted:
            kept.add(track_id)
        per_track.append({
            "local_track_id": track_id,
            "samples": len(values),
            "supported_samples": supported,
            "support_ratio": round(ratio, 6),
            "mean_turf_score": round(float(np.mean(values)), 6),
            "geometry_support_ratio": round(geometry_ratio, 6),
            "accepted": accepted,
        })

    filtered = [
        row for row_index, row in enumerate(rows)
        if int(row[1]) in kept and row_index in geometry_supported_rows
    ]
    report = {
        "method": "track_vote_of_field_geometry_and_bottom_patch_turf_support",
        "min_detection_score": min_detection_score,
        "min_track_ratio": min_track_ratio,
        "min_track_samples": min_track_samples,
        "min_foot_y_ratio": min_foot_y_ratio,
        "min_geometry_ratio": min_geometry_ratio,
        "field_geometry_mode": geometry.mode,
        "input_detections": len(rows),
        "output_detections": len(filtered),
        "geometry_rejected_detections": len(rows) - len(geometry_supported_rows),
        "input_tracklets": len(scores),
        "kept_tracklets": len(kept),
        "rejected_tracklets": len(scores) - len(kept),
        "tracks": sorted(per_track, key=lambda item: item["local_track_id"]),
    }
    return FieldFilterResult(filtered, kept, report)


def restrict_tracklet_frames(tracklets: dict, detections: Iterable[tuple]) -> None:
    """Synchronize tracklet frame sets after a detection-domain filter."""
    frames: dict[int, set[int]] = defaultdict(set)
    for frame, track_id, *_ in detections:
        frames[int(track_id)].add(int(frame))
    for track_id, tracklet in tracklets.items():
        accepted = frames.get(int(track_id), set())
        tracklet["frames"] = accepted
        tracklet["first"] = min(accepted) if accepted else None
        tracklet["last"] = max(accepted) if accepted else None
        samples = [sample for sample in tracklet.get("team_feature_samples", []) if sample[0] in accepted]
        tracklet["team_feature_samples"] = samples
        tracklet["team_feature"] = aggregate_features([sample[1] for sample in samples])
