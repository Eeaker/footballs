from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mode_split.local_change import detect_local_feature_transitions


def block(vector: list[float], count: int, noise: float = 0.0) -> np.ndarray:
    rows = np.repeat(np.asarray(vector, np.float32)[None, :], count, axis=0)
    if noise:
        rng = np.random.default_rng(20260824)
        rows += rng.normal(0.0, noise, rows.shape).astype(np.float32)
    rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)
    return rows


def test_persistent_track_local_change_is_split() -> None:
    features = np.vstack([block([1, 0, 0], 20, .01), block([0, 1, 0], 25, .01)])
    cuts = detect_local_feature_transitions(np.arange(1, 46), features, window=8)
    assert len(cuts) == 1
    assert 19 <= cuts[0].frame <= 22


def test_normal_within_track_variation_does_not_split() -> None:
    features = block([1, .2, .1], 80, .035)
    assert detect_local_feature_transitions(np.arange(1, 81), features, window=8) == []


def test_short_excursion_returning_to_track_baseline_is_suppressed() -> None:
    features = np.vstack([
        block([1, 0, 0], 30, .01), block([0, 1, 0], 20, .01), block([1, 0, 0], 40, .01),
    ])
    assert detect_local_feature_transitions(
        np.arange(1, 91), features, window=8, return_horizon_frames=30,
    ) == []


def test_a_different_nonplayer_baseline_is_not_globally_relabelled() -> None:
    black_track = block([.05, .05, 1], 60, .02)
    red_track = block([.4, .8, .1], 60, .02)
    assert detect_local_feature_transitions(np.arange(1, 61), black_track, window=8) == []
    assert detect_local_feature_transitions(np.arange(1, 61), red_track, window=8) == []
