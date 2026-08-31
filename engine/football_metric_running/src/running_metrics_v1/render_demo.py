"""Render metric-running results with an on-frame mini pitch."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

def color_for_id(global_id: int) -> tuple[int, int, int]:
    hue = (global_id * 47) % 180
    pixel = np.uint8([[[hue, 210, 245]]])
    return tuple(int(v) for v in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])


def read_timeseries(path: Path) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {
                "global_id": int(row["global_id"]),
                "proc_idx": int(row["proc_idx"]),
                "foot_x_px": float(row["foot_x_px"]),
                "foot_y_px": float(row["foot_y_px"]),
                "x_m_smooth": float(row["x_m_smooth"]),
                "y_m_smooth": float(row["y_m_smooth"]),
                "speed_mps": None if row["speed_mps"] == "" else float(row["speed_mps"]),
            }
            by_frame[parsed["proc_idx"]].append(parsed)
    return dict(by_frame)


def draw_pitch_inset(
    frame: np.ndarray,
    rows: list[dict],
    bounds: dict,
) -> None:
    # Keep the inset in the same visual orientation as the source image:
    # smaller pitch-y values are farther/up-screen and larger values are
    # nearer/down-screen.  This is deliberately a screen-view convention,
    # not the usual Cartesian convention where positive y points upward.
    inset_w, inset_h = 450, 250
    x0, y0 = frame.shape[1] - inset_w - 20, 20
    x1, y1 = x0 + inset_w, y0 + inset_h
    cv2.rectangle(frame, (x0, y0), (x1, y1), (30, 105, 30), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)

    xmin, xmax = float(bounds["x_min"]), float(bounds["x_max"])
    ymin, ymax = float(bounds["y_min"]), float(bounds["y_max"])

    def to_inset(x_m: float, y_m: float) -> tuple[int, int]:
        px = x0 + int(round((x_m - xmin) / max(xmax - xmin, 1e-9) * inset_w))
        py = y0 + int(round((y_m - ymin) / max(ymax - ymin, 1e-9) * inset_h))
        return px, py

    # Static pitch markings on the metric plane.  The centre circle radius is
    # the provisional 3 m value used by this first Demo calibration.
    line_color = (235, 235, 235)
    centre_x = (xmin + xmax) / 2.0
    centre_y = (ymin + ymax) / 2.0
    cv2.line(frame, to_inset(centre_x, ymin), to_inset(centre_x, ymax), line_color, 2)
    centre_px = to_inset(centre_x, centre_y)
    radius_px = max(1, int(round(3.0 / (xmax - xmin) * inset_w)))
    cv2.circle(frame, centre_px, radius_px, line_color, 2)
    cv2.circle(frame, centre_px, 3, line_color, -1)

    # Player mapping: points only; no accumulated movement trails.
    for row in rows:
        point = to_inset(row["x_m_smooth"], row["y_m_smooth"])
        color = color_for_id(row["global_id"])
        cv2.circle(frame, point, 6, color, -1)
        cv2.circle(frame, point, 7, (255, 255, 255), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render metric-running Demo video")
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--timeseries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-proc", type=int)
    parser.add_argument("--end-proc", type=int)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    video_path = Path(calibration["video"])
    rows_by_frame = read_timeseries(args.timeseries)
    start = int(calibration["valid_start_proc"])
    end = int(calibration["valid_end_proc"])
    if args.start_proc is not None:
        start = max(start, args.start_proc)
    if args.end_proc is not None:
        end = min(end, args.end_proc)
    if end < start:
        parser.error("requested render interval is empty")
    stride = int(calibration["vid_stride"])
    fps = float(calibration["video_metadata"]["proc_fps"])
    width = int(calibration["video_metadata"]["frame_width"])
    height = int(calibration["video_metadata"]["frame_height"])

    capture = cv2.VideoCapture(str(video_path))
    raw_start = (start + 1) * stride - 1
    raw_end = (end + 1) * stride - 1
    capture.set(cv2.CAP_PROP_POS_FRAMES, raw_start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create video: {args.output}")

    is_dynamic = calibration.get("camera_model") == "dynamic_per_frame_homography"
    bounds = calibration["field_bounds_m"]
    raw_idx = raw_start - 1
    while raw_idx < raw_end:
        ok, frame = capture.read()
        if not ok:
            break
        raw_idx += 1
        if (raw_idx + 1) % stride != 0:
            continue
        proc_idx = (raw_idx + 1) // stride - 1
        if proc_idx < start or proc_idx > end:
            continue
        rows = rows_by_frame.get(proc_idx, [])
        for row in rows:
            point = (int(round(row["foot_x_px"])), int(round(row["foot_y_px"])))
            color = color_for_id(row["global_id"])
            cv2.circle(frame, point, 6, color, -1)
            speed = row["speed_mps"]
            label = f"ID {row['global_id']}" if speed is None else f"ID {row['global_id']} {speed:.2f} m/s"
            cv2.putText(frame, label, (point[0] + 8, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        draw_pitch_inset(frame, rows, bounds)
        mode_label = (
            "Dynamic-H Demo: per-frame pitch registration"
            if is_dynamic
            else "Fixed-H Demo: valid only while camera pose/zoom stay unchanged"
        )
        cv2.putText(frame, mode_label,
                    (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, f"proc_idx={proc_idx} time={proc_idx / fps:.3f}s",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    capture.release()
    writer.release()
    print(f"Demo video: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
