"""Evaluate pure-rotation image registration against one calibrated frame.

The camera is assumed to keep a fixed optical centre and fixed zoom while
panning/tilting.  Under that condition, two views are related by an image
homography even for background objects at different depths.
"""

from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path

import cv2
import numpy as np


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_index}")
    return frame


def features(frame: np.ndarray, scale: float = 0.5):
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    mask = np.zeros_like(gray)
    # Ignore sky and the near foreground, concentrating on buildings, fence,
    # far pitch markings and other scene-static structure.
    mask[int(gray.shape[0] * 0.16):int(gray.shape[0] * 0.82), :] = 255
    detector = cv2.SIFT_create(nfeatures=4500, contrastThreshold=0.025)
    keypoints, descriptors = detector.detectAndCompute(gray, mask)
    return keypoints, descriptors, small


def estimate_current_to_reference(
    current_features,
    reference_features,
    scale: float = 0.5,
) -> tuple[np.ndarray | None, dict]:
    cur_kp, cur_desc, _ = current_features
    ref_kp, ref_desc, _ = reference_features
    if cur_desc is None or ref_desc is None:
        return None, {"matches": 0, "inliers": 0, "inlier_ratio": 0.0}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(cur_desc, ref_desc, k=2)
    good = [a for a, b in pairs if a.distance < 0.72 * b.distance]
    if len(good) < 12:
        return None, {"matches": len(good), "inliers": 0, "inlier_ratio": 0.0}
    src = np.float32([cur_kp[m.queryIdx].pt for m in good]) / scale
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good]) / scale
    matrix, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    stats = {
        "matches": len(good),
        "inliers": inliers,
        "inlier_ratio": inliers / len(good) if good else 0.0,
    }
    if matrix is None or inliers < 15 or stats["inlier_ratio"] < 0.25:
        return None, stats
    matrix = matrix / matrix[2, 2]
    return matrix, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--reference-frame", required=True, type=int)
    parser.add_argument("--sample-seconds", required=True, nargs="+", type=float)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--reference-calibration", type=Path)
    parser.add_argument("--overlay-dir", type=Path)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    ref_frame = read_frame(capture, args.reference_frame)
    ref_features = features(ref_frame)
    metric_h = None
    if args.reference_calibration:
        calibration = json.loads(args.reference_calibration.read_text(encoding="utf-8"))
        metric_h = np.asarray(calibration["H_image_to_pitch_m"], dtype=np.float64)
    if args.overlay_dir:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seconds in args.sample_seconds:
        frame_index = int(round(seconds * fps))
        frame = read_frame(capture, frame_index)
        matrix, stats = estimate_current_to_reference(features(frame), ref_features)
        rows.append({
            "seconds": seconds,
            "frame_index": frame_index,
            "accepted": matrix is not None,
            **stats,
            "H_current_to_reference": "" if matrix is None else repr(matrix.tolist()),
        })
        if matrix is not None and metric_h is not None and args.overlay_dir:
            current_to_metric = metric_h @ matrix
            metric_to_current = np.linalg.inv(current_to_metric)

            def project_world(points: np.ndarray) -> np.ndarray:
                src = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
                return cv2.perspectiveTransform(src, metric_to_current).reshape(-1, 2)

            overlay = frame.copy()
            # Draw only metric markings whose dimensions are currently known.
            boundary = project_world(np.array([[0, 0], [45, 0], [45, 25], [0, 25]]))
            cv2.polylines(overlay, [np.round(boundary).astype(np.int32)], True, (0, 255, 255), 3)
            halfway = project_world(np.array([[22.5, 0], [22.5, 25]]))
            cv2.polylines(overlay, [np.round(halfway).astype(np.int32)], False, (255, 0, 255), 3)
            angles = np.linspace(0, 2 * np.pi, 121)
            circle_world = np.column_stack([
                22.5 + 3.0 * np.cos(angles),
                12.5 + 3.0 * np.sin(angles),
            ])
            circle = project_world(circle_world)
            cv2.polylines(overlay, [np.round(circle).astype(np.int32)], True, (255, 0, 255), 3)
            cv2.putText(
                overlay,
                f"t={seconds:.1f}s inliers={stats['inliers']} ratio={stats['inlier_ratio']:.2f}",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
            )
            cv2.imwrite(str(args.overlay_dir / f"{seconds:07.1f}s.jpg"), overlay)
        print(rows[-1])
    capture.release()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
