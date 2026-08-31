from __future__ import annotations

import json

from analysis_lib.geometry import HomographyProvider, _portable_basename


def test_portable_basename_handles_windows_artifact_on_linux():
    assert _portable_basename(r"C:\data\足球视频7月24日.mp4") == "足球视频7月24日.mp4"


def test_explicitly_rejected_dynamic_frame_is_not_filled_from_nearest_matrix(tmp_path):
    calibration = tmp_path / "dynamic.json"
    calibration.write_text(json.dumps({
        "camera_model": "dynamic_per_frame_homography",
        "video": "synthetic.mp4",
        "vid_stride": 1,
        "validation": {"passed": True},
        "field_bounds_m": {"x_min": 0, "x_max": 45, "y_min": 0, "y_max": 25, "margin_m": .5},
        "frames": [
            {"proc_idx": 0, "accepted": True, "H_image_to_pitch_m": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            {"proc_idx": 1, "accepted": False, "H_image_to_pitch_m": None, "reason": "registration_sample_gap"},
            {"proc_idx": 2, "accepted": True, "H_image_to_pitch_m": [[1, 0, 2], [0, 1, 0], [0, 0, 1]]},
        ],
    }), encoding="utf-8")
    provider = HomographyProvider(calibration)
    assert provider.at_processed_frame(0, 1) is not None
    assert provider.at_processed_frame(1, 1) is None
    assert provider.at_processed_frame(2, 1) is not None
    assert provider.in_field((0, 0)) is True
    assert provider.in_field((45.5, 25.5)) is True
    assert provider.in_field((46, 10)) is False
