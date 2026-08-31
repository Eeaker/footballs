"""Calculate per-player metric running summaries from Stage 3 MOT output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from running_metrics_v1.metrics import calculate_running_metrics, timeseries_as_dicts
from running_metrics_v1.mot import inspect_records, read_mot
from running_metrics_v1.dynamic_calibration import homographies_from_calibration


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Metric running v1 from MOT + calibration")
    parser.add_argument("--mot", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--median-window", type=int, default=11)
    parser.add_argument("--high-speed-threshold", type=float, default=4.5)
    parser.add_argument("--allow-failed-calibration", action="store_true")
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    validation_passed = bool(calibration.get("validation", {}).get("passed", False))
    if not validation_passed and not args.allow_failed_calibration:
        parser.error("calibration did not pass its independent length check")
    proc_fps = float(calibration["video_metadata"]["proc_fps"])
    records = read_mot(args.mot)
    if calibration.get("camera_model") == "dynamic_per_frame_homography":
        homography = homographies_from_calibration(calibration)
        if not homography:
            parser.error("dynamic calibration contains no accepted frame homographies")
    else:
        homography = np.asarray(calibration["H_image_to_pitch_m"], dtype=np.float64)
    summary, timeseries, quality = calculate_running_metrics(
        records=records,
        homography=homography,
        proc_fps=proc_fps,
        valid_start_proc=int(calibration["valid_start_proc"]),
        valid_end_proc=int(calibration["valid_end_proc"]),
        field_bounds=calibration.get("field_bounds_m"),
        median_window=args.median_window,
        high_speed_threshold_mps=args.high_speed_threshold,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "player_running_summary.csv", summary)
    write_csv(args.outdir / "player_running_timeseries.csv", timeseries_as_dicts(timeseries))
    quality["mot_preflight"] = inspect_records(
        records, calibration["video_metadata"].get("proc_total_frames")
    )
    quality["calibration_file"] = str(args.calibration.resolve())
    quality["calibration_validation_passed"] = validation_passed
    (args.outdir / "running_quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Players: {len(summary)}")
    print(f"Summary: {args.outdir / 'player_running_summary.csv'}")
    print(f"Time series: {args.outdir / 'player_running_timeseries.csv'}")
    print(f"Quality report: {args.outdir / 'running_quality_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
