"""Build per-frame metric homographies from multiple manually calibrated anchor views.

Designed for a fixed optical centre / pan-tilt camera. Each sampled frame is
registered to the temporally nearest calibrated anchor, so large rotations do
not have to match one distant reference view for the whole match.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from running_metrics_v1.evaluate_rotation_registration import estimate_current_to_reference, features, read_frame


def _load_anchor(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not bool((data.get("validation") or {}).get("passed", False)):
        raise ValueError(f"anchor not independently validated: {path}")
    H = np.asarray(data["H_image_to_pitch_m"], dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError(f"invalid anchor homography: {path}")
    return {
        "path": str(path),
        "proc_idx": int(data["calibration_proc_idx"]),
        "H": H / H[2, 2],
        "data": data,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build multi-anchor rotating-camera dynamic calibration")
    parser.add_argument("--anchor", action="append", required=True, type=Path, help="Validated reference calibration JSON; repeat 1-4 times")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--max-interpolation-gap", type=int, default=30)
    parser.add_argument("--limit-frames", type=int)
    args = parser.parse_args()
    if args.sample_step < 1:
        parser.error("sample-step must be positive")
    if not 1 <= len(args.anchor) <= 6:
        parser.error("anchor count must be 1-6")

    anchors = sorted((_load_anchor(path) for path in args.anchor), key=lambda a: a["proc_idx"])
    base = anchors[0]["data"]
    video_path = Path(base["video"])
    for anchor in anchors[1:]:
        if Path(anchor["data"]["video"]).resolve() != video_path.resolve():
            raise ValueError("anchors refer to different videos")
        if anchor["data"].get("field_bounds_m") != base.get("field_bounds_m"):
            raise ValueError("anchor field bounds differ")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.limit_frames is not None:
        total = min(total, args.limit_frames)

    # Pre-compute SIFT features for each calibrated anchor view.
    for anchor in anchors:
        if anchor["proc_idx"] >= total:
            raise ValueError(f"anchor outside video: {anchor['proc_idx']}")
        frame = read_frame(capture, anchor["proc_idx"])
        anchor["features"] = features(frame)

    def nearest_anchor(frame_idx: int) -> dict:
        return min(anchors, key=lambda a: abs(a["proc_idx"] - frame_idx))

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    sampled_indices: list[int] = []
    sampled_metric_h: list[np.ndarray] = []
    sample_quality: list[dict] = []
    started = time.time()
    frame_index = -1
    anchor_frames = {a["proc_idx"]: a for a in anchors}
    while frame_index + 1 < total:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % args.sample_step != 0 and frame_index not in anchor_frames:
            continue
        anchor = nearest_anchor(frame_index)
        if frame_index == anchor["proc_idx"]:
            current_to_anchor = np.eye(3, dtype=np.float64)
            stats = {"matches": 0, "inliers": 0, "inlier_ratio": 1.0}
        else:
            current_to_anchor, stats = estimate_current_to_reference(features(frame), anchor["features"])
        accepted = current_to_anchor is not None
        row = {"proc_idx": frame_index, "accepted": accepted, "anchor_proc_idx": anchor["proc_idx"], **stats}
        sample_quality.append(row)
        if accepted:
            metric_h = anchor["H"] @ current_to_anchor
            metric_h /= metric_h[2, 2]
            sampled_indices.append(frame_index)
            sampled_metric_h.append(metric_h)
        if len(sample_quality) % 300 == 0:
            elapsed = max(time.time() - started, 1e-9)
            rate = max(frame_index / elapsed, 1e-9)
            eta = (total - frame_index - 1) / rate
            print(f"frame={frame_index}/{total-1} samples={len(sample_quality)} accepted={len(sampled_indices)} eta_s={eta:.1f}", flush=True)
    capture.release()

    if len(sampled_indices) < 2:
        raise RuntimeError("not enough accepted registration samples")
    order = np.argsort(sampled_indices)
    sample_x = np.asarray(sampled_indices, dtype=np.int64)[order]
    sample_h = np.asarray(sampled_metric_h, dtype=np.float64)[order]

    all_x = np.arange(total, dtype=np.int64)
    interpolated = np.empty((total, 3, 3), dtype=np.float64)
    for row in range(3):
        for col in range(3):
            interpolated[:, row, col] = np.interp(all_x, sample_x, sample_h[:, row, col])
    interpolated /= interpolated[:, 2:3, 2:3]

    insertion = np.searchsorted(sample_x, all_x)
    before_pos = np.clip(insertion - 1, 0, len(sample_x) - 1)
    after_pos = np.clip(insertion, 0, len(sample_x) - 1)
    before = sample_x[before_pos]
    after = sample_x[after_pos]
    bracket_gap = after - before
    edge_distance = np.minimum(np.abs(all_x - before), np.abs(after - all_x))
    accepted_frame = (bracket_gap <= args.max_interpolation_gap) | ((bracket_gap == 0) & (edge_distance <= args.max_interpolation_gap))

    frames = []
    for index in range(total):
        if not accepted_frame[index]:
            frames.append({"proc_idx": index, "accepted": False, "H_image_to_pitch_m": None, "reason": "registration_sample_gap"})
        else:
            frames.append({"proc_idx": index, "accepted": True, "H_image_to_pitch_m": interpolated[index].tolist()})

    result = {
        "schema_version": 2,
        "camera_model": "dynamic_per_frame_homography",
        "camera_assumption": "fixed_optical_centre_pan_tilt",
        "warning": "Invalid if the camera translates materially or zoom/focal length changes substantially.",
        "video": str(video_path),
        "video_metadata": base.get("video_metadata", {}),
        "vid_stride": 1,
        "field_bounds_m": base.get("field_bounds_m"),
        "valid_start_proc": 0,
        "valid_end_proc": total - 1,
        "frames": frames,
        "anchors": [
            {
                "proc_idx": a["proc_idx"],
                "source": a["path"],
                "validation": a["data"].get("validation"),
                "calibration_point_reprojection_error_m": a["data"].get("calibration_point_reprojection_error_m"),
            } for a in anchors
        ],
        "dynamic_registration": {
            "method": "nearest-anchor SIFT + USAC_MAGSAC registration, direct metric-H interpolation",
            "anchor_proc_indices": [a["proc_idx"] for a in anchors],
            "sample_step_frames": args.sample_step,
            "max_interpolation_gap_frames": args.max_interpolation_gap,
            "accepted_frame_count": int(accepted_frame.sum()),
            "rejected_frame_count": int((~accepted_frame).sum()),
            "accepted_sample_count": len(sampled_indices),
            "sample_count": len(sample_quality),
            "samples": sample_quality,
        },
        "validation": {"passed": False, "note": "Final pass/fail is set by product service after coverage + anchor validation checks."},
        "provenance": {"source": "football_insight_multi_anchor_dynamic_calibration_v2"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"dynamic calibration: {args.output}")
    print(f"anchors: {len(anchors)}; accepted frames: {int(accepted_frame.sum())}/{total}; accepted samples: {len(sampled_indices)}/{len(sample_quality)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
