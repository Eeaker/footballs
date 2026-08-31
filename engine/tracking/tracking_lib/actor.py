from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class BoxObservation:
    frame: int
    global_id: int
    x: float
    y: float
    w: float
    h: float


def interpolate_ball(
    ball_pos: Mapping[int, tuple[float, float]], total_frames: int, max_gap: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate only short gaps and mark frames backed by a valid segment."""
    bx = np.full(total_frames, np.nan, dtype=np.float64)
    by = np.full(total_frames, np.nan, dtype=np.float64)
    for frame, point in ball_pos.items():
        if 0 <= int(frame) < total_frames:
            bx[int(frame)], by[int(frame)] = float(point[0]), float(point[1])

    observed = np.flatnonzero(np.isfinite(bx) & np.isfinite(by))
    reliable = np.zeros(total_frames, dtype=bool)
    if observed.size == 0:
        return bx, by, reliable
    reliable[observed] = True
    for left, right in zip(observed[:-1], observed[1:]):
        gap = int(right - left)
        if gap <= 1 or gap > max_gap:
            continue
        alpha = np.arange(1, gap, dtype=np.float64) / gap
        bx[left + 1 : right] = bx[left] + alpha * (bx[right] - bx[left])
        by[left + 1 : right] = by[left] + alpha * (by[right] - by[left])
        reliable[left:right + 1] = True
    return bx, by, reliable


def build_global_boxes(
    detections: Iterable[tuple], local_to_global: Mapping[int, int]
) -> dict[int, list[BoxObservation]]:
    by_frame: dict[int, list[BoxObservation]] = defaultdict(list)
    for frame, local_id, x, y, w, h, *_ in detections:
        if local_id not in local_to_global:
            continue
        by_frame[int(frame)].append(
            BoxObservation(
                frame=int(frame),
                global_id=int(local_to_global[local_id]),
                x=float(x), y=float(y), w=float(w), h=float(h),
            )
        )
    return dict(by_frame)


def _point_to_actor_distance(ball_x: float, ball_y: float, box: BoxObservation) -> float:
    """Scale-normalized distance to a lower-body interaction region.

    The interaction region includes the bbox and a small margin below/around it.
    Zero means the ball is inside that region. Values near one are roughly one
    player-height away and should normally be considered weak evidence.
    """
    left = box.x - 0.20 * box.w
    right = box.x + 1.20 * box.w
    top = box.y + 0.35 * box.h
    bottom = box.y + 1.20 * box.h
    dx = max(left - ball_x, 0.0, ball_x - right)
    dy = max(top - ball_y, 0.0, ball_y - bottom)
    scale = max(box.h, 1.0)
    return math.hypot(dx, dy) / scale


def attribute_actor(
    event_frame: int,
    fps: float,
    boxes_by_frame: Mapping[int, list[BoxObservation]],
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    reliable_ball: np.ndarray,
    *,
    pre_seconds: float = 0.55,
    post_seconds: float = 0.15,
    max_normalized_distance: float = 1.25,
    top_k: int = 3,
) -> dict:
    """Rank likely event actors without pretending the score is calibrated.

    Kicks normally precede the acceleration peak, hence the asymmetric window.
    The returned attribution_score is a ranking score, not a probability.
    """
    start = max(0, int(round(event_frame - pre_seconds * fps)))
    end = min(len(ball_x) - 1, int(round(event_frame + post_seconds * fps)))
    evidence: dict[int, tuple[float, int, float]] = {}

    for frame in range(start, end + 1):
        if frame >= len(reliable_ball) or not reliable_ball[frame]:
            continue
        if not np.isfinite(ball_x[frame]) or not np.isfinite(ball_y[frame]):
            continue
        time_penalty = abs(frame - event_frame) / max(fps * 0.5, 1.0)
        for box in boxes_by_frame.get(frame, []):
            distance = _point_to_actor_distance(ball_x[frame], ball_y[frame], box)
            rank_cost = distance + 0.18 * time_penalty
            old = evidence.get(box.global_id)
            if old is None or rank_cost < old[0]:
                evidence[box.global_id] = (rank_cost, frame, distance)

    ranked = sorted(evidence.items(), key=lambda item: (item[1][0], item[0]))
    candidates = []
    for global_id, (cost, frame, distance) in ranked[: max(1, top_k)]:
        candidates.append({
            "global_id": int(global_id),
            "attribution_score": round(float(math.exp(-cost)), 4),
            "evidence_frame_proc": int(frame),
            "normalized_ball_distance": round(float(distance), 4),
        })

    if not candidates:
        return {
            "primary_global_id": None,
            "actor_attribution_status": "review",
            "actor_attribution_reason": "no_reliable_ball_player_overlap",
            "actor_candidates": [],
        }

    best = candidates[0]
    second_score = candidates[1]["attribution_score"] if len(candidates) > 1 else 0.0
    margin = best["attribution_score"] - second_score
    acceptable = best["normalized_ball_distance"] <= max_normalized_distance
    confident = acceptable and (margin >= 0.12 or best["normalized_ball_distance"] <= 0.15)
    return {
        "primary_global_id": best["global_id"] if acceptable else None,
        "actor_attribution_status": "auto" if confident else "review",
        "actor_attribution_reason": (
            "nearest_player_before_event_peak" if acceptable
            else "nearest_player_too_far_from_ball"
        ),
        "actor_candidates": candidates,
    }
