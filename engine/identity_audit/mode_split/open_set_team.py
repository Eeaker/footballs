from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TeamModeModel:
    centers: np.ndarray
    radii: np.ndarray
    separation: float
    source_mode_sizes: tuple[int, int]
    appearance_modes: int


@dataclass(frozen=True)
class TeamSwitch:
    frame: int
    before_mode: int
    after_mode: int
    before_support: int
    after_support: int


def learn_open_set_team_modes(
    features: np.ndarray, *, appearance_modes: int = 6, sample_limit: int = 30000,
    seed: int = 0,
) -> TeamModeModel:
    """Learn two dominant team modes while leaving all smaller modes open-set."""
    values = np.asarray(features, np.float32)
    if len(values) < appearance_modes * 10:
        raise ValueError("not enough accepted torso features")
    stride = max(1, len(values) // sample_limit)
    sample = values[::stride][:sample_limit]
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-5)
    _, labels, centers = cv2.kmeans(
        sample, appearance_modes, None, criteria, 8, cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.ravel()
    sizes = np.bincount(labels, minlength=appearance_modes)
    selected = np.argsort(-sizes)[:2]
    team_centers = centers[selected].astype(np.float32)
    separation = float(np.linalg.norm(team_centers[0] - team_centers[1]))
    if min(sizes[selected]) < .12 * len(sample) or separation < .35:
        raise ValueError("two stable dominant team modes were not discovered")
    radii = []
    for team_index, source_index in enumerate(selected):
        distances = np.linalg.norm(sample[labels == source_index] - team_centers[team_index], axis=1)
        radii.append(float(np.quantile(distances, .97)))
    return TeamModeModel(
        centers=team_centers,
        radii=np.asarray(radii, np.float32),
        separation=separation,
        source_mode_sizes=(int(sizes[selected[0]]), int(sizes[selected[1]])),
        appearance_modes=appearance_modes,
    )


def assign_open_set_team_modes(features: np.ndarray, model: TeamModeModel) -> np.ndarray:
    values = np.asarray(features, np.float32)
    distances = np.linalg.norm(values[:, None, :] - model.centers[None, :, :], axis=2)
    labels = distances.argmin(axis=1).astype(np.int8)
    ordered = np.sort(distances, axis=1)
    margin = ordered[:, 1] - ordered[:, 0]
    reject = distances[np.arange(len(values)), labels] > model.radii[labels]
    reject |= margin < .08 * model.separation
    labels[reject] = -1
    return labels


def detect_persistent_team_switches(
    frames: np.ndarray, labels: np.ndarray, *, window: int = 30,
    minimum_confident: int = 15, purity: float = .80, collapse_frames: int = 30,
) -> list[TeamSwitch]:
    frames = np.asarray(frames, np.int64)
    labels = np.asarray(labels, np.int8)
    if len(frames) != len(labels):
        raise ValueError("frames/labels shape mismatch")
    raw: list[TeamSwitch] = []
    for index in range(window, len(labels) - window + 1):
        left, right = labels[index - window:index], labels[index:index + window]
        left = left[left >= 0]
        right = right[right >= 0]
        if len(left) < minimum_confident or len(right) < minimum_confident:
            continue
        left_counts = np.bincount(left, minlength=2)
        right_counts = np.bincount(right, minlength=2)
        before, after = int(left_counts.argmax()), int(right_counts.argmax())
        if before == after:
            continue
        if left_counts[before] / len(left) < purity or right_counts[after] / len(right) < purity:
            continue
        first_after = np.flatnonzero(labels[index:index + window] == after)
        raw.append(TeamSwitch(
            frame=int(frames[index + int(first_after[0])]), before_mode=before,
            after_mode=after, before_support=int(left_counts[before]),
            after_support=int(right_counts[after]),
        ))
    collapsed: list[TeamSwitch] = []
    for item in raw:
        if collapsed and item.frame - collapsed[-1].frame <= collapse_frames:
            if item.before_support + item.after_support > collapsed[-1].before_support + collapsed[-1].after_support:
                collapsed[-1] = item
        else:
            collapsed.append(item)

    # With exactly two team states, accepted transitions must alternate.  A
    # repeated A->B candidate without a credible B->A in between is another
    # view of the same change, usually separated by an unknown/occluded band.
    consistent: list[TeamSwitch] = []
    current_mode: int | None = None
    for item in collapsed:
        if current_mode is None:
            consistent.append(item)
            current_mode = item.after_mode
        elif item.before_mode == current_mode and item.after_mode != current_mode:
            consistent.append(item)
            current_mode = item.after_mode
    return consistent
