"""Interactive four-point pitch calibration with independent length checks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from running_metrics_v1.homography import (
    point_reprojection_errors_m,
    project_points,
    solve_image_to_world,
    validate_segments,
)


def raw_frame_index_for_proc(proc_idx: int, vid_stride: int) -> int:
    return (proc_idx + 1) * vid_stride - 1


def read_video_frame(video: Path, proc_idx: int, vid_stride: int) -> tuple[np.ndarray, dict]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    metadata = {
        "raw_fps": float(capture.get(cv2.CAP_PROP_FPS) or 30.0),
        "raw_total_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "frame_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "frame_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    raw_idx = raw_frame_index_for_proc(proc_idx, vid_stride)
    if raw_idx < 0 or raw_idx >= metadata["raw_total_frames"]:
        capture.release()
        raise ValueError(f"calibration raw frame {raw_idx} is outside the video")
    capture.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not decode raw frame {raw_idx}")
    metadata["raw_frame_index"] = raw_idx
    metadata["proc_fps"] = metadata["raw_fps"] / vid_stride
    metadata["proc_total_frames"] = metadata["raw_total_frames"] // vid_stride
    return frame, metadata


def collect_clicks(frame: np.ndarray, count: int, title: str) -> list[list[float]]:
    max_width, max_height = 1600, 900
    scale = min(1.0, max_width / frame.shape[1], max_height / frame.shape[0])
    display = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()
    points: list[list[float]] = []

    def redraw() -> None:
        canvas = display.copy()
        for index, (x, y) in enumerate(points, start=1):
            dx, dy = int(round(x * scale)), int(round(y * scale))
            cv2.circle(canvas, (dx, dy), 7, (0, 0, 255), -1)
            cv2.putText(canvas, str(index), (dx + 9, dy - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(canvas, f"Left click: add ({len(points)}/{count}) | Right click: undo | Enter: confirm",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 255), 2)
        cv2.imshow(title, canvas)

    def mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < count:
            points.append([x / scale, y / scale])
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            redraw()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, mouse)
    redraw()
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 10) and len(points) == count:
            break
        if key == 27:
            cv2.destroyWindow(title)
            raise KeyboardInterrupt("calibration cancelled")
    cv2.destroyWindow(title)
    return points


def build_overlay(frame: np.ndarray, calibration: dict) -> np.ndarray:
    overlay = frame.copy()
    image_points = calibration["image_points"]
    for index, (x, y) in enumerate(image_points, start=1):
        point = (int(round(x)), int(round(y)))
        cv2.circle(overlay, point, 8, (0, 0, 255), -1)
        cv2.putText(overlay, f"C{index}", (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for index, segment in enumerate(calibration["validation_segments"], start=1):
        p1, p2 = segment["image_points"]
        a = tuple(int(round(v)) for v in p1)
        b = tuple(int(round(v)) for v in p2)
        cv2.line(overlay, a, b, (0, 255, 255), 3)
        cv2.putText(overlay, f"V{index}: {segment['known_length_m']:.2f}m",
                    a, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    bounds = calibration["field_bounds_m"]
    world_corners = [
        [bounds["x_min"], bounds["y_min"]], [bounds["x_max"], bounds["y_min"]],
        [bounds["x_max"], bounds["y_max"]], [bounds["x_min"], bounds["y_max"]],
    ]
    try:
        image_corners = project_points(world_corners, np.linalg.inv(np.asarray(calibration["H_image_to_pitch_m"])))
        polygon = np.round(image_corners).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [polygon], True, (0, 255, 0), 3)
    except ValueError:
        pass
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Four-point image-to-pitch calibration")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--proc-frame", required=True, type=int)
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--valid-start-proc", type=int)
    parser.add_argument("--valid-end-proc", type=int)
    parser.add_argument("--field-width-m", required=True, type=float)
    parser.add_argument("--field-height-m", required=True, type=float)
    parser.add_argument("--field-margin-m", type=float, default=0.5)
    parser.add_argument("--validation-tolerance-m", type=float, default=0.5)
    parser.add_argument("--validation-count", type=int, default=1)
    parser.add_argument("--points-json", type=Path,
                        help="Non-interactive point input; see calibration_points.example.json")
    args = parser.parse_args()

    if args.proc_frame < 0 or args.vid_stride < 1:
        parser.error("proc-frame must be >=0 and vid-stride must be >=1")
    if args.field_width_m <= 0 or args.field_height_m <= 0:
        parser.error("field dimensions must be positive")
    if args.validation_count < 1:
        parser.error("at least one independent validation segment is required")

    frame, video_meta = read_video_frame(args.video, args.proc_frame, args.vid_stride)
    if args.points_json:
        point_data = json.loads(args.points_json.read_text(encoding="utf-8"))
        image_points = point_data["image_points"]
        world_points = point_data["world_points_m"]
        validation_segments = point_data["validation_segments"]
    else:
        print("Click four ground-plane reference points. Their order only needs to match the coordinates entered next.")
        image_points = collect_clicks(frame, 4, "Calibration points")
        world_points = []
        for index in range(4):
            value = input(f"World X,Y in meters for calibration point {index + 1}: ")
            world_points.append([float(v.strip()) for v in value.split(",")])
        validation_segments = []
        for index in range(args.validation_count):
            points = collect_clicks(frame, 2, f"Independent validation segment {index + 1}")
            known = float(input(f"Known length in meters for validation segment {index + 1}: "))
            validation_segments.append({
                "name": f"segment_{index + 1}",
                "image_points": points,
                "known_length_m": known,
            })

    matrix = solve_image_to_world(image_points, world_points)
    validations = validate_segments(matrix, validation_segments, args.validation_tolerance_m)
    reprojection = point_reprojection_errors_m(matrix, image_points, world_points)
    start = args.proc_frame if args.valid_start_proc is None else args.valid_start_proc
    end = args.proc_frame if args.valid_end_proc is None else args.valid_end_proc
    if start < 0 or end < start or end >= video_meta["proc_total_frames"]:
        parser.error("invalid H validity interval")

    result = {
        "schema_version": 1,
        "camera_model": "single_fixed_homography",
        "warning": "H is valid only while camera pose and zoom remain unchanged.",
        "video": str(args.video.resolve()),
        "video_metadata": video_meta,
        "vid_stride": args.vid_stride,
        "calibration_proc_idx": args.proc_frame,
        "valid_start_proc": start,
        "valid_end_proc": end,
        "image_points": image_points,
        "world_points_m": world_points,
        "H_image_to_pitch_m": matrix.tolist(),
        "field_bounds_m": {
            "x_min": 0.0, "x_max": args.field_width_m,
            "y_min": 0.0, "y_max": args.field_height_m,
            "margin_m": args.field_margin_m,
        },
        "calibration_point_reprojection_error_m": {
            "max": float(reprojection.max()),
            "mean": float(reprojection.mean()),
        },
        "validation_segments": validation_segments,
        "validation": {
            "tolerance_m": args.validation_tolerance_m,
            "passed": bool(validations) and all(v.passed for v in validations),
            "results": [asdict(value) for value in validations],
        },
        "provenance": {
            "projection_convention": "TVCalib pixel2world example (MIT); see THIRD_PARTY_NOTICES.md",
            "homography_solver": "OpenCV getPerspectiveTransform/findHomography",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    overlay_path = args.output.with_name(args.output.stem + "_overlay.jpg")
    cv2.imwrite(str(overlay_path), build_overlay(frame, result))
    print(f"Calibration: {args.output}")
    print(f"Overlay: {overlay_path}")
    for validation in validations:
        print(f"{validation.name}: measured={validation.measured_length_m:.3f}m "
              f"error={validation.absolute_error_m:.3f}m passed={validation.passed}")
    return 0 if result["validation"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

