import cv2
import numpy as np

from tracking_lib.field_filter import filter_tracklets_by_turf, turf_support_score


def test_turf_support_accepts_green_foot_patch():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[60:, :] = (40, 150, 40)
    assert turf_support_score(frame, (35, 30, 20, 40)) > 0.5


def test_turf_support_rejects_concrete_foot_patch():
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    assert turf_support_score(frame, (35, 30, 20, 40)) == 0.0


def test_footpoint_roi_rejects_stand_detection_but_keeps_field_detection(tmp_path):
    video = tmp_path / "one_frame.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (100, 100))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:] = (40, 150, 40)
    writer.write(frame)
    writer.release()
    detections = [
        (0, 1, 10.0, 5.0, 10.0, 20.0, 0.9, 0),
        (0, 2, 50.0, 50.0, 10.0, 30.0, 0.9, 0),
    ]
    result = filter_tracklets_by_turf(
        video, detections, min_track_samples=1, min_foot_y_ratio=0.32
    )
    assert [int(row[1]) for row in result.detections] == [2]
    assert result.report["geometry_rejected_detections"] == 1
