"""Build per-frame metric homographies for a fixed-centre rotating camera."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from running_metrics_v1.evaluate_rotation_registration import (
    estimate_current_to_reference,
    features,
    read_frame,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--max-interpolation-gap", type=int, default=30)
    parser.add_argument("--limit-frames", type=int)
    args = parser.parse_args()
    if args.sample_step < 1:
        parser.error("sample-step must be positive")

    base = json.loads(args.reference_calibration.read_text(encoding="utf-8"))
    video_path = Path(base["video"])
    reference_index = int(base["calibration_proc_idx"])
    metric_reference_h = np.asarray(base["H_image_to_pitch_m"], dtype=np.float64)

    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.limit_frames is not None:
        total = min(total, args.limit_frames)
    reference = read_frame(capture, reference_index)
    reference_features = features(reference)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    sampled_indices: list[int] = []
    sampled_matrices: list[np.ndarray] = []
    sample_quality: list[dict] = []
    started = time.time()
    frame_index = -1
    while frame_index + 1 < total:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % args.sample_step != 0 and frame_index != reference_index:
            continue
        if frame_index == reference_index:
            matrix = np.eye(3, dtype=np.float64)
            stats = {"matches": 0, "inliers": 0, "inlier_ratio": 1.0}
        else:
            matrix, stats = estimate_current_to_reference(
                features(frame), reference_features
            )
        accepted = matrix is not None
        sample_quality.append({
            "proc_idx": frame_index,
            "accepted": accepted,
            **stats,
        })
        if accepted:
            sampled_indices.append(frame_index)
            sampled_matrices.append(matrix / matrix[2, 2])
        if len(sample_quality) % 300 == 0:
            elapsed = max(time.time() - started, 1e-9)
            rate = frame_index / elapsed
            eta = (total - frame_index - 1) / max(rate, 1e-9)
            accepted_count = sum(item["accepted"] for item in sample_quality)
            print(
                f"frame={frame_index}/{total - 1} samples={len(sample_quality)} "
                f"accepted={accepted_count} eta_s={eta:.1f}",
                flush=True,
            )
    capture.release()

    if len(sampled_indices) < 2:
        raise RuntimeError("not enough accepted registration samples")
    order = np.argsort(sampled_indices)
    sample_x = np.asarray(sampled_indices, dtype=np.int64)[order]
    sample_h = np.asarray(sampled_matrices, dtype=np.float64)[order]

    all_x = np.arange(total, dtype=np.int64)
    interpolated = np.empty((total, 3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            interpolated[:, row, column] = np.interp(
                all_x, sample_x, sample_h[:, row, column]
            )
    interpolated /= interpolated[:, 2:3, 2:3]

    insertion = np.searchsorted(sample_x, all_x)
    before_pos = np.clip(insertion - 1, 0, len(sample_x) - 1)
    after_pos = np.clip(insertion, 0, len(sample_x) - 1)
    before = sample_x[before_pos]
    after = sample_x[after_pos]
    bracket_gap = after - before
    edge_distance = np.minimum(np.abs(all_x - before), np.abs(after - all_x))
    accepted_frame = (bracket_gap <= args.max_interpolation_gap) | (
        (bracket_gap == 0) & (edge_distance <= args.max_interpolation_gap)
    )

    frames = []
    for index in range(total):
        if not accepted_frame[index]:
            frames.append({
                "proc_idx": index,
                "accepted": False,
                "H_image_to_pitch_m": None,
                "reason": "registration_sample_gap",
            })
            continue
        metric_h = metric_reference_h @ interpolated[index]
        metric_h /= metric_h[2, 2]
        frames.append({
            "proc_idx": index,
            "accepted": True,
            "H_image_to_pitch_m": metric_h.tolist(),
        })

    result = dict(base)
    result.pop("H_image_to_pitch_m", None)
    result["camera_model"] = "dynamic_per_frame_homography"
    result["warning"] = (
        "Derived from a fixed-optical-centre rotation assumption; invalid if "
        "the camera translates or zoom changes."
    )
    result["valid_start_proc"] = 0
    result["valid_end_proc"] = total - 1
    result["frames"] = frames
    result["dynamic_registration"] = {
        "method": "SIFT + USAC_MAGSAC current-frame to calibrated-reference homography",
        "reference_proc_idx": reference_index,
        "sample_step_frames": args.sample_step,
        "max_interpolation_gap_frames": args.max_interpolation_gap,
        "accepted_frame_count": int(accepted_frame.sum()),
        "rejected_frame_count": int((~accepted_frame).sum()),
        "accepted_sample_count": len(sampled_indices),
        "sample_count": len(sample_quality),
        "samples": sample_quality,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"dynamic calibration: {args.output}")
    print(
        f"accepted frames: {int(accepted_frame.sum())}/{total}; "
        f"accepted samples: {len(sampled_indices)}/{len(sample_quality)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
