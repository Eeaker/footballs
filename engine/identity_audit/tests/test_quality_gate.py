import numpy as np

from mode_split.quality_gate import intersection_over_target, quality_gated_torso_feature, torso_rect


def test_torso_geometry_matches_tracking_crop():
    assert torso_rect((100, 50, 40, 100)) == (109.6, 60.0, 130.4, 108.0)


def test_overlap_is_relative_to_target_torso():
    assert intersection_over_target((0, 0, 10, 10), (5, 0, 10, 10)) == .5


def test_overlapped_torso_becomes_unknown_not_a_feature():
    frame = np.full((100, 100, 3), (255, 0, 0), np.uint8)
    feature, quality = quality_gated_torso_feature(
        frame, (20, 10, 30, 70), [(25, 15, 30, 70)], minimum_sharpness=0,
    )
    assert feature is None
    assert quality.reason == "person_overlap"
