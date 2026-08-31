"""Build a dynamic calibration document from per-frame pitch correspondences."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from running_metrics_v1.dynamic_calibration import solve_frame_calibrations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare per-frame homographies for moving-camera footage"
    )
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--base-calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-reprojection-error-m", type=float, default=0.5)
    parser.add_argument("--min-accepted-ratio", type=float, default=0.8)
    args = parser.parse_args()

    if not 0 < args.min_accepted_ratio <= 1:
        parser.error("min-accepted-ratio must be in (0, 1]")
    base = json.loads(args.base_calibration.read_text(encoding="utf-8"))
    payload = json.loads(args.observations.read_text(encoding="utf-8"))
    observations = payload["frames"] if isinstance(payload, dict) else payload
    matrices, frame_results = solve_frame_calibrations(
        observations, args.max_reprojection_error_m
    )
    if not frame_results:
        parser.error("observations contain no frames")

    accepted_ratio = len(matrices) / len(frame_results)
    scale_validation_passed = bool(base.get("validation", {}).get("passed", False))
    frames = []
    for result in frame_results:
        item = asdict(result)
        item["H_image_to_pitch_m"] = item.pop("homography")
        frames.append(item)
    output = {
        "schema_version": 1,
        "camera_model": "dynamic_per_frame_homography",
        "warning": "Only frames with an accepted H contribute to running metrics.",
        "video": base["video"],
        "video_metadata": base["video_metadata"],
        "vid_stride": base["vid_stride"],
        "valid_start_proc": min(result.proc_idx for result in frame_results),
        "valid_end_proc": max(result.proc_idx for result in frame_results),
        "field_bounds_m": base["field_bounds_m"],
        "frames": frames,
        "validation": {
            "passed": scale_validation_passed and accepted_ratio >= args.min_accepted_ratio,
            "metric_scale_validation_passed": scale_validation_passed,
            "accepted_frames": len(matrices),
            "total_observed_frames": len(frame_results),
            "accepted_ratio": accepted_ratio,
            "min_accepted_ratio": args.min_accepted_ratio,
            "max_reprojection_error_m": args.max_reprojection_error_m,
        },
        "provenance": {
            "implementation": "independent implementation in running_metrics_v1",
            "architecture_reference": "https://github.com/abdullahtarek/basketball_analysis",
            "source_copy": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dynamic calibration: {args.output}")
    print(f"Accepted frames: {len(matrices)}/{len(frame_results)} ({accepted_ratio:.1%})")
    return 0 if output["validation"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

