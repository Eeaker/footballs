import numpy as np

from onboard.video_health import classify_camera_motion, uniform_frame_indices


def test_uniform_indices_are_bounded_and_unique():
    values = uniform_frame_indices(100, 50)
    assert len(values) == len(set(values)) == 50
    assert 0 <= values[0] < values[-1] < 100


def test_fixed_camera_classification():
    measurements = [{"translation_px": .2, "rotation_deg": .01, "inlier_ratio": .9, "residual_px": .2}] * 8
    motion, usable, mode, _ = classify_camera_motion(measurements, 1000)
    assert (motion, usable, mode) == ("fixed", True, "static")


def test_unstable_camera_disables_automatic_dynamic_h():
    measurements = [{"translation_px": 30, "rotation_deg": 1, "inlier_ratio": .2, "residual_px": 8}] * 8
    motion, usable, mode, _ = classify_camera_motion(measurements, 1000)
    assert (motion, usable, mode) == ("handheld_translate", False, "manual_keyframes")


def test_intermittent_horizontal_pan_is_not_misclassified_as_fixed():
    stationary = [{"translation_px": .4, "rotation_deg": .01, "inlier_ratio": .90, "residual_px": .3}] * 7
    panning = [{"translation_px": 24, "rotation_deg": .02, "inlier_ratio": .86, "residual_px": .5}] * 3
    motion, usable, mode, _ = classify_camera_motion(stationary + panning, 1000)
    assert (motion, usable, mode) == ("pan_rotate", True, "dynamic_keyframes")
