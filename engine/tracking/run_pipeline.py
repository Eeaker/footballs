from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tracking_lib.config import get_value, load_config, resolve_config_path
from tracking_lib.metric_motion import export_metric_motion


ROOT = Path(__file__).resolve().parent
FINAL_TRACKING_FILES = {
    "tracking_vis.mp4": "tracking_vis.mp4",
    "tracking_mot.txt": "tracking_mot.txt",
    "event_index.json": "events.json",
    "event_index.csv": "events.csv",
    "global_id_summary.json": "global_id_summary.json",
    "ball_positions_observed.csv": "ball_positions_observed.csv",
    "field_filter_report.json": "field_filter_report.json",
    "tracking_run_metadata.json": "tracking_run_metadata.json",
}

OPTIONAL_TRACKING_FILES = ("metric_positions.csv", "metric_motion_summary.json")


def normalize_thread_environment(environment: dict[str, str] | None = None) -> dict[str, str]:
    """修正 AutoDL 可能注入的非法线程数（如 OMP_NUM_THREADS=0）。"""
    target = os.environ if environment is None else environment
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        raw = target.get(key)
        if raw is None:
            continue
        try:
            valid = int(raw) >= 1
        except (TypeError, ValueError):
            valid = False
        if not valid:
            target[key] = "1"
    return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], log) -> None:
    rendered = " ".join(command)
    print(f"\n$ {rendered}", flush=True)
    log.write(f"\n$ {rendered}\n")
    log.flush()
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, env=child_env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
    return_code = process.wait()
    log.flush()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tracking V3工程交付：单入口完整管线")
    parser.add_argument("--config", type=Path, help="onboard.py生成的新场地配置")
    parser.add_argument("--video", type=Path, help="显式指定时覆盖配置中的视频")
    parser.add_argument("--weights", type=Path, help="显式指定时覆盖配置中的权重")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--vid-stride", type=int)
    return parser.parse_args()


def main() -> None:
    normalize_thread_environment()
    args = parse_args()
    config = load_config(args.config) if args.config else {}
    video = args.video.resolve() if args.video else resolve_config_path(config, "video.path")
    weights = args.weights.resolve() if args.weights else resolve_config_path(config, "detector.weights")
    if video is None or weights is None:
        raise ValueError("必须通过命令行或配置提供 video 和 weights")
    output = args.output.resolve()
    imgsz = args.imgsz if args.imgsz is not None else int(get_value(config, "detector.imgsz", 1280))
    vid_stride = args.vid_stride if args.vid_stride is not None else int(get_value(config, "tracker.vid_stride", 1))
    tracker = resolve_config_path(config, "tracker.config_file", ROOT / "config/botsort_buffer.yaml")
    expected_players = int(get_value(config, "scene.expected_on_field_players", 16))
    team_clusters = int(get_value(config, "identity.team_clusters", 2))
    edge_margin = float(get_value(config, "events.edge_margin_seconds", 20))
    pre_sec = float(get_value(config, "highlights.seconds_before_event", 15))
    post_sec = float(get_value(config, "highlights.seconds_after_event", 15))
    event_count = int(get_value(config, "highlights.event_count", 5))
    for path, label in ((video, "video"), (weights, "weights")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录必须不存在或为空，避免覆盖既有结果: {output}")
    output.mkdir(parents=True, exist_ok=True)
    work = output / ".work"
    tracking_work = work / "tracking"
    tracking_final = output / "tracking"
    tracking_work.mkdir(parents=True)
    started = time.time()

    with (output / "pipeline.log").open("w", encoding="utf-8") as log:
        tracking_command = [
            sys.executable, str(ROOT / "run_tracking.py"),
            "--tracking-core", str(ROOT / "tracking_core.py"),
            "--video", str(video), "--weights", str(weights),
            "--outdir", str(tracking_work), "--device", args.device,
            "--imgsz", str(imgsz), "--vid_stride", str(vid_stride),
            "--conf", str(get_value(config, "detector.confidence", .25)),
            "--tracker-config", str(tracker),
            "--max_ids", str(int(get_value(config, "identity.max_ids", expected_players))),
            "--min_track_frames", str(get_value(config, "identity.min_track_frames", 10)),
            "--min_presence_ratio", str(get_value(config, "identity.min_presence_ratio", .005)),
            "--expected_on_field_players", str(expected_players),
            "--team_clusters", str(team_clusters), "--edge_margin", str(edge_margin),
            "--event_percentile", str(get_value(config, "events.percentile", 92)),
            "--event_min_gap", str(get_value(config, "events.minimum_gap_seconds", 2)),
            "--min_turf_score", str(get_value(config, "field_filter.min_turf_score", .15)),
            "--min_track_turf_ratio", str(get_value(config, "field_filter.min_track_turf_ratio", .25)),
            "--min_foot_y_ratio", str(get_value(config, "field_filter.min_foot_y_ratio", .32)),
            "--pre_sec", str(pre_sec), "--post_sec", str(post_sec),
            "--n_clips", str(event_count),
        ]
        if args.config:
            tracking_command.extend([
                "--onboard-config", str(args.config.resolve()),
                "--min_geometry_ratio", str(get_value(config, "field_filter.min_geometry_ratio", .60)),
            ])
        if not bool(get_value(config, "field_filter.enabled", True)):
            tracking_command.append("--disable_field_filter")
        run(tracking_command, log)
        calibration = get_value(config, "calibration", {})
        if calibration:
            metadata = json.loads((tracking_work / "tracking_run_metadata.json").read_text(encoding="utf-8"))
            export_metric_motion(tracking_work / "tracking_mot.txt", tracking_work, calibration,
                                 float(metadata["processed_fps"]), vid_stride)
        tracking_final.mkdir(parents=True, exist_ok=True)
        for source_name, final_name in FINAL_TRACKING_FILES.items():
            source = tracking_work / source_name
            if not source.exists():
                raise FileNotFoundError(source)
            os.replace(source, tracking_final / final_name)
        for source_name in OPTIONAL_TRACKING_FILES:
            source = tracking_work / source_name
            if source.exists():
                os.replace(source, tracking_final / source_name)


    shutil.rmtree(work)
    manifest = {
        "pipeline": "tracking_pipeline_v4",
        "status": "complete",
        "input_video": str(video),
        "input_video_sha256": sha256(video),
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "elapsed_seconds": round(time.time() - started, 3),
        "policy": {
            "players_on_field": expected_players,
            "teams": team_clusters,
            "global_id_policy": "conservative_keep_all_candidates_then_human_identity_review",
            "evaluation": "canonical_tracking_outputs_only",
        },
        "key_artifact_hashes": {
            str(path.relative_to(output)): sha256(path) for path in [
                tracking_final / "tracking_mot.txt",
                tracking_final / "events.json",
                tracking_final / "global_id_summary.json",
            ]
        },
    }
    if args.config:
        manifest["onboard_config"] = str(args.config.resolve())
        manifest["onboard_config_sha256"] = sha256(args.config.resolve())
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output),
                      "elapsed_seconds": manifest["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
