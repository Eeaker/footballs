from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class LocalFeatureTransition:
    frame: int
    feature_distance: float
    adaptive_ratio: float
    left_variability: float
    right_variability: float
    before_center: np.ndarray = field(repr=False, compare=False)
    after_center: np.ndarray = field(repr=False, compare=False)


def _center(rows: np.ndarray) -> np.ndarray:
    center = np.median(rows.astype(np.float32), axis=0)
    return center / max(float(np.linalg.norm(center)), 1e-9)


def _variability(rows: np.ndarray, center: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(rows.astype(np.float32) - center[None, :], axis=1)))


def detect_local_feature_transitions(
    frames: np.ndarray, features: np.ndarray, *, window: int = 8,
    absolute_distance: float = .35, variability_ratio: float = 2.2,
    minimum_variability: float = .035, maximum_side_span_frames: int | None = None,
    return_horizon_frames: int = 30, return_baseline_distance: float = .20,
) -> list[LocalFeatureTransition]:
    """Detect persistent appearance changes using only one track's box history.

    The test is self-normalising: the before/after difference must exceed both
    an absolute distance and the normal within-window variation.  No global
    colour cluster, semantic team label, or manually supplied colour is used.
    """
    frames = np.asarray(frames, np.int64)
    features = np.asarray(features, np.float32)
    if features.ndim != 2 or len(frames) != len(features):
        raise ValueError("frames/features shape mismatch")
    if len(frames) < 2 * window or window < 2:
        return []
    order = np.argsort(frames)
    frames, features = frames[order], features[order]
    valid = np.isfinite(features).all(axis=1) & (np.linalg.norm(features, axis=1) > 1e-6)
    frames, features = frames[valid], features[valid]
    if len(frames) < 2 * window:
        return []
    max_span = maximum_side_span_frames or window * 3

    raw: list[LocalFeatureTransition] = []
    for index in range(window, len(frames) - window + 1):
        left_frames, right_frames = frames[index - window:index], frames[index:index + window]
        if left_frames[-1] - left_frames[0] > max_span or right_frames[-1] - right_frames[0] > max_span:
            continue
        left, right = features[index - window:index], features[index:index + window]
        before, after = _center(left), _center(right)
        distance = float(np.linalg.norm(before - after))
        left_var, right_var = _variability(left, before), _variability(right, after)
        adaptive = max(left_var + right_var, minimum_variability)
        ratio = distance / adaptive
        if distance < absolute_distance or ratio < variability_ratio:
            continue
        raw.append(LocalFeatureTransition(
            frame=int(right_frames[0]), feature_distance=distance, adaptive_ratio=ratio,
            left_variability=left_var, right_variability=right_var,
            before_center=before, after_center=after,
        ))

    # A single physical change triggers adjacent window boundaries.  Keep the
    # strongest boundary from each contiguous candidate band.
    collapsed: list[LocalFeatureTransition] = []
    for candidate in raw:
        if collapsed and candidate.frame - collapsed[-1].frame <= window:
            if candidate.adaptive_ratio > collapsed[-1].adaptive_ratio:
                collapsed[-1] = candidate
        else:
            collapsed.append(candidate)

    # Suppress a short appearance excursion that returns to the original local
    # baseline.  This covers temporary overlap, shadow, and partial torso crops.
    accepted: list[LocalFeatureTransition] = []
    index = 0
    while index < len(collapsed):
        current = collapsed[index]
        if index + 1 < len(collapsed):
            following = collapsed[index + 1]
            returned = float(np.linalg.norm(current.before_center - following.after_center))
            if (following.frame - current.frame <= return_horizon_frames
                    and returned <= return_baseline_distance):
                index += 2
                continue
        accepted.append(current)
        index += 1
    return accepted
