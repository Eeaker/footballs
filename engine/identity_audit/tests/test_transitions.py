from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mode_split.transitions import ModeObservation, detect_persistent_transitions


def observations(ranges: list[tuple[int, int, int | None, float]]) -> list[ModeObservation]:
    result: list[ModeObservation] = []
    for start, end, mode, confidence in ranges:
        result.extend(ModeObservation(frame, mode, confidence) for frame in range(start, end + 1))
    return result


def test_splits_after_short_occlusion_when_stable_mode_changes() -> None:
    rows = observations([
        (1, 20, 0, .95),
        (21, 23, None, 0.0),
        (24, 45, 1, .93),
    ])
    cuts = detect_persistent_transitions(rows, window=8, minimum_evidence=5, purity=0.75)
    assert [cut.frame for cut in cuts] == [24]
    assert cuts[0].before_mode == 0
    assert cuts[0].after_mode == 1


def test_does_not_split_on_one_frame_colour_flicker() -> None:
    rows = observations([(1, 30, 0, .95)])
    rows[14] = ModeObservation(15, 1, .99)
    assert detect_persistent_transitions(rows, window=8, minimum_evidence=5, purity=0.75) == []


def test_does_not_split_when_mode_is_unchanged_across_gap() -> None:
    rows = observations([
        (1, 20, 2, .91),
        (21, 25, None, 0.0),
        (26, 45, 2, .92),
    ])
    assert detect_persistent_transitions(rows, window=8, minimum_evidence=5, purity=0.75) == []


def test_low_confidence_observations_are_ignored() -> None:
    rows = observations([(1, 20, 0, .95), (21, 40, 1, .20), (41, 60, 0, .95)])
    assert detect_persistent_transitions(
        rows, window=8, minimum_evidence=5, purity=0.75, minimum_confidence=.35,
    ) == []


def test_short_stable_excursion_that_returns_to_original_mode_is_not_a_cut() -> None:
    rows = observations([(1, 30, 0, .95), (31, 50, 1, .95), (51, 90, 0, .95)])
    assert detect_persistent_transitions(
        rows, window=8, minimum_evidence=5, purity=0.75, reversal_horizon_frames=30,
    ) == []
