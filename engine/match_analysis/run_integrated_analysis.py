"""One-click match analysis analysis + exact legacy running metrics + standalone pitch video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from analysis_lib.pipeline import PipelineConfig, run_analysis
from analysis_lib.pitch_render import render_pitch_video


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_mot(root: Path) -> Path:
    for relative in ("tracking_mot.txt", "tracking/tracking_mot.txt"):
        path = root / relative
        if path.is_file():
            return path
    raise FileNotFoundError(f"tracking MOT not found under {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="一键生成球权/主动传球网络/米制跑动/独立球场视频")
    parser.add_argument("--tracking-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="必须不存在的新根目录")
    parser.add_argument("--team-map", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--running-src", type=Path,
                        default=Path(__file__).resolve().parent.parent / "football_metric_running" / "src",
                        help="旧 football_metric_running/src；原模块将被直接调用")
    parser.add_argument("--team-clusters", type=int, choices=(2, 3), default=2)
    parser.add_argument("--team-samples-per-id", type=int, default=12)
    parser.add_argument("--max-ball-gap-frames", type=int, default=2)
    parser.add_argument("--max-transfer-gap-seconds", type=float, default=1.5)
    parser.add_argument("--min-pass-displacement-m", type=float, default=.5)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"输出根目录必须不存在，避免覆盖: {output}")
    running_src = args.running_src.resolve()
    legacy_entry = running_src / "running_metrics_v1" / "calculate_running.py"
    if not legacy_entry.is_file():
        raise FileNotFoundError(f"旧米制跑动源码不存在: {legacy_entry}")
    output.mkdir(parents=True)
    started = time.time()

    analysis_dir = output / "analysis"
    report = run_analysis(
        tracking_dir=args.tracking_dir, calibration=args.calibration,
        output=analysis_dir, video=args.video, team_map_path=args.team_map,
        annotations=args.annotations,
        config=PipelineConfig(
            team_clusters=args.team_clusters,
            team_samples_per_id=args.team_samples_per_id,
            max_ball_gap_frames=args.max_ball_gap_frames,
            max_transfer_gap_seconds=args.max_transfer_gap_seconds,
            min_pass_displacement_m=args.min_pass_displacement_m,
        ),
    )

    running_dir = output / "metric_running"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(running_src) + os.pathsep + environment.get("PYTHONPATH", "")
    command = [
        sys.executable, "-m", "running_metrics_v1.calculate_running",
        "--mot", str(resolve_mot(args.tracking_dir.resolve())),
        "--calibration", str(args.calibration.resolve()),
        "--outdir", str(running_dir),
    ]
    subprocess.run(command, check=True, env=environment)

    pitch_video = output / "metric_pitch_with_possession_and_passes.mp4"
    render_report = render_pitch_video(
        calibration_path=args.calibration,
        timeseries_path=running_dir / "player_running_timeseries.csv",
        possession_path=analysis_dir / "possession_frame_evidence.csv",
        transitions_path=analysis_dir / "possession_transitions.csv",
        passes_path=analysis_dir / "pass_events.csv",
        team_map_path=analysis_dir / "player_team_map.csv",
        output_path=pitch_video,
    )
    manifest = {
        "pipeline": "match_analysis_metric_pitch_v2",
        "status": report["status"],
        "elapsed_seconds": round(time.time() - started, 3),
        "legacy_running_source": str(running_src),
        "legacy_calculate_running_sha256": sha256(legacy_entry),
        "analysis_report": report,
        "render_report": render_report,
        "key_artifacts": {
            "analysis/possession_transitions.csv": sha256(analysis_dir / "possession_transitions.csv"),
            "analysis/pass_events.csv": sha256(analysis_dir / "pass_events.csv"),
            "metric_running/player_running_summary.csv": sha256(running_dir / "player_running_summary.csv"),
            "metric_running/player_running_timeseries.csv": sha256(running_dir / "player_running_timeseries.csv"),
            "metric_pitch_with_possession_and_passes.mp4": sha256(pitch_video),
        },
    }
    (output / "integrated_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
