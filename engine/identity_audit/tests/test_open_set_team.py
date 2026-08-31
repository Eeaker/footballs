import numpy as np

from mode_split.open_set_team import detect_persistent_team_switches


def test_unknown_occlusion_does_not_create_a_switch():
    frames = np.arange(1, 101)
    labels = np.asarray([0] * 35 + [-1] * 30 + [0] * 35, np.int8)
    assert detect_persistent_team_switches(frames, labels, window=15, minimum_confident=8) == []


def test_sustained_team_change_cuts_at_first_new_confident_frame():
    frames = np.arange(1, 101)
    labels = np.asarray([0] * 45 + [-1] * 5 + [1] * 50, np.int8)
    cuts = detect_persistent_team_switches(frames, labels, window=20, minimum_confident=10)
    assert len(cuts) == 1
    assert cuts[0].frame == 51


def test_repeated_same_direction_candidate_is_not_emitted_twice():
    frames = np.arange(1, 181)
    # The short ambiguous island cannot establish a reverse switch, so the
    # later 0->1 evidence must not create a second copy of that transition.
    labels = np.asarray([0] * 50 + [1] * 50 + [-1] * 20 + [0] * 8 + [1] * 52, np.int8)
    cuts = detect_persistent_team_switches(
        frames, labels, window=20, minimum_confident=10, purity=.8, collapse_frames=20,
    )
    assert [(item.before_mode, item.after_mode) for item in cuts] == [(0, 1)]
