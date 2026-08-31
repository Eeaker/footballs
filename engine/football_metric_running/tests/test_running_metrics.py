import json
from pathlib import Path

import numpy as np

from running_metrics_v1.homography import (
    project_points,
    solve_image_to_world,
    validate_segments,
)
from running_metrics_v1.metrics import calculate_running_metrics, median_smooth
from running_metrics_v1.mot import MotDetection, inspect_records, read_mot
from running_metrics_v1.dynamic_calibration import solve_frame_calibrations


def detection(frame_mot, gid, foot_x, foot_y, confidence=0.9):
    return MotDetection(
        proc_idx=frame_mot - 1,
        global_id=gid,
        x=foot_x - 1.0,
        y=foot_y - 2.0,
        w=2.0,
        h=2.0,
        confidence=confidence,
    )


def test_four_point_homography_and_independent_segment_validation():
    image = [[100, 100], [500, 100], [450, 400], [150, 400]]
    world = [[0, 0], [40, 0], [40, 20], [0, 20]]
    matrix = solve_image_to_world(image, world)
    midpoint_pixels = project_points([[20, 10]], np.linalg.inv(matrix))[0]
    mapped = project_points([midpoint_pixels], matrix)[0]
    assert np.allclose(mapped, [20, 10], atol=1e-5)
    p1, p2 = project_points([[10, 10], [30, 10]], np.linalg.inv(matrix))
    result = validate_segments(matrix, [{
        "name": "holdout", "image_points": [p1, p2], "known_length_m": 20.0
    }])[0]
    assert result.passed
    assert result.absolute_error_m < 1e-5


def test_mot_frame_is_converted_from_one_based_to_zero_based(tmp_path):
    path = tmp_path / "mot.txt"
    path.write_text("1,7,10,20,4,8,0.9,-1,-1,-1\n", encoding="utf-8")
    row = read_mot(path)[0]
    assert row.proc_idx == 0
    assert row.foot_x == 12
    assert row.foot_y == 28


def test_median_smoothing_suppresses_single_frame_spike():
    values = np.zeros(21)
    values[10] = 100.0
    smoothed = median_smooth(values, window=11)
    assert np.all(smoothed == 0)


def test_constant_six_mps_metrics_follow_requested_definitions():
    fps = 30.0
    records = [detection(i + 1, 3, i * 6.0 / fps, 5.0) for i in range(61)]
    summary, rows, _ = calculate_running_metrics(
        records, np.eye(3), fps, 0, 60,
        field_bounds={"x_min": 0, "x_max": 20, "y_min": 0, "y_max": 10},
    )
    assert len(summary) == 1
    assert np.isclose(summary[0]["total_distance_m"], 12.0, atol=1e-3)
    assert np.isclose(summary[0]["high_speed_distance_m"], 12.0, atol=1e-3)
    assert np.isclose(summary[0]["peak_speed_mps_p95"], 6.0, atol=1e-3)
    assert len(rows) == 61


def test_gap_is_split_and_never_bridged():
    fps = 10.0
    first = [detection(i + 1, 1, i * 0.2, 1.0) for i in range(11)]
    second = [detection(i + 101, 1, 100 + i * 0.2, 1.0) for i in range(11)]
    summary, _, quality = calculate_running_metrics(
        first + second, np.eye(3), fps, 0, 110,
        field_bounds={"x_min": -1, "x_max": 200, "y_min": 0, "y_max": 2},
    )
    assert np.isclose(summary[0]["total_distance_m"], 4.0, atol=1e-3)
    assert summary[0]["continuous_segments"] == 2
    assert quality["gap_policy"].startswith("only delta_frame=1")


def test_duplicate_identity_frame_is_excluded_and_reported():
    records = [detection(i + 1, 4, i * 0.1, 1.0) for i in range(15)]
    records.append(detection(8, 4, 50.0, 1.0))
    summary, _, quality = calculate_running_metrics(
        records, np.eye(3), 10.0, 0, 14,
        field_bounds={"x_min": -1, "x_max": 100, "y_min": 0, "y_max": 2},
    )
    assert summary[0]["collision_frames"] == 1
    assert "identity_collision" in summary[0]["quality_flags"]
    assert quality["identities"][0]["collision_proc_indices"] == [7]
    preflight = inspect_records(records, total_proc_frames=15)
    assert preflight["duplicate_identity_frame_keys"] == 1


def test_short_segments_are_excluded_from_metrics():
    records = [detection(i + 1, 2, i * 0.1, 1.0) for i in range(10)]
    summary, rows, _ = calculate_running_metrics(
        records, np.eye(3), 10.0, 0, 9,
        field_bounds={"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 2},
    )
    assert summary[0]["total_distance_m"] == 0
    assert summary[0]["short_segment_frames"] == 10
    assert rows == []


def test_per_frame_homography_removes_camera_pan_from_stationary_player():
    fps = 10.0
    records = []
    observations = []
    world_corners = [[0, 0], [20, 0], [20, 10], [0, 10]]
    for proc_idx in range(15):
        pan = proc_idx * 3.0
        image_corners = [
            [100 + pan, 100], [500 + pan, 100],
            [500 + pan, 300], [100 + pan, 300],
        ]
        observations.append({
            "proc_idx": proc_idx,
            "image_points": image_corners,
            "world_points_m": world_corners,
        })
        # The stationary world point (10, 5) moves in image space with the pan.
        records.append(detection(proc_idx + 1, 8, 300 + pan, 200))

    matrices, report = solve_frame_calibrations(observations)
    summary, rows, quality = calculate_running_metrics(
        records, matrices, fps, 0, 14,
        field_bounds={"x_min": 0, "x_max": 20, "y_min": 0, "y_max": 10},
    )
    assert all(item.accepted for item in report)
    assert np.isclose(summary[0]["total_distance_m"], 0.0, atol=1e-6)
    assert len(rows) == 15
    assert quality["calibration_policy"].startswith("per-frame H")


def test_missing_dynamic_homography_splits_track_and_is_reported():
    records = [detection(i + 1, 9, i * 0.1, 1.0) for i in range(23)]
    matrices = {i: np.eye(3) for i in range(23) if i != 11}
    summary, _, _ = calculate_running_metrics(
        records, matrices, 10.0, 0, 22,
        field_bounds={"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 2},
    )
    assert summary[0]["continuous_segments"] == 2
    assert summary[0]["missing_calibration_frames"] == 1
    assert "missing_frame_calibration" in summary[0]["quality_flags"]
