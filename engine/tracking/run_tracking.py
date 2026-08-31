from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from tracking_lib.actor import attribute_actor, build_global_boxes, interpolate_ball
from tracking_lib.config import get_value, load_config
from tracking_lib.field_filter import filter_tracklets_by_turf, restrict_tracklet_frames


def load_tracking_core(path: Path):
    spec = importlib.util.spec_from_file_location("legacy_tracking_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载旧管线: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tracking隔离运行：复用tracking 流程并增强事件ID归属")
    parser.add_argument("--tracking-core", type=Path, default=Path("tracking_core.py"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--onboard-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--video", required=True)
    parser.add_argument("--outdir", default="outputs/tracking")
    parser.add_argument("--weights", default="models/yolov8x.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--tracker-config", default="config/botsort_buffer.yaml",
        help="tracking BoT-SORT：保持检测阈值，仅延长遮挡缓冲（A/B验证后关闭通用ReID）",
    )
    parser.add_argument("--vid_stride", type=int, default=1)
    parser.add_argument("--reid_stride", type=int, default=10)
    parser.add_argument("--reid_interval_sec", type=float, default=0.33)
    parser.add_argument(
        "--max_ids", type=int, default=16,
        help="本场16名球员的重关联目标；默认不再作为渲染硬截断",
    )
    parser.add_argument(
        "--drop_extra_clusters", action="store_true",
        help="恢复旧行为：只保留前 max_ids 个簇（默认关闭）",
    )
    parser.add_argument("--expected_on_field_players", type=int, default=16)
    parser.add_argument("--team_clusters", type=int, default=2)
    parser.add_argument(
        "--no_render_unassigned", action="store_true",
        help="不在检查视频中显示尚未进入稳定簇的短轨迹",
    )
    parser.add_argument("--disable_field_filter", action="store_true")
    parser.add_argument("--min_turf_score", type=float, default=0.15)
    parser.add_argument("--min_track_turf_ratio", type=float, default=0.25)
    parser.add_argument(
        "--min_foot_y_ratio", type=float, default=0.32,
        help="tracking图像空间场地ROI：球员脚点至少位于画面高度的该比例以下",
    )
    parser.add_argument("--min_geometry_ratio", type=float, default=0.0)
    parser.add_argument("--team_prototype_margin", type=float, default=0.03)
    parser.add_argument("--min_track_frames", type=int, default=10)
    parser.add_argument("--min_presence_ratio", type=float, default=0.005)
    parser.add_argument("--allow_presence_backfill", action="store_true")
    parser.add_argument("--merge_floor", type=float, default=0.10)
    parser.add_argument("--wa", type=float, default=0.6)
    parser.add_argument("--wc", type=float, default=0.4)
    parser.add_argument("--color_min", type=float, default=0.15)
    parser.add_argument("--pre_sec", type=float, default=15.0,
                        help="集锦候选事件点之前保留15秒")
    parser.add_argument("--post_sec", type=float, default=15.0,
                        help="集锦候选事件点之后保留15秒")
    parser.add_argument("--event_percentile", type=float, default=92.0)
    parser.add_argument("--event_min_gap", type=float, default=2.0)
    parser.add_argument("--edge_margin", type=float, default=20.0, help="tracking前20秒镜头抖动，默认排除")
    parser.add_argument("--ball_max_gap", type=int, default=30)
    parser.add_argument("--n_clips", type=int, default=5,
                        help="集锦默认5个事件，每个事件前后各15秒，共约150秒")
    parser.add_argument("--actor_pre_sec", type=float, default=0.55)
    parser.add_argument("--actor_post_sec", type=float, default=0.15)
    parser.add_argument("--actor_max_distance", type=float, default=1.25)
    args = parser.parse_args()
    core = load_tracking_core(args.tracking_core)
    core.validate_args(parser, args)
    if args.actor_pre_sec < 0 or args.actor_post_sec < 0 or args.actor_max_distance <= 0:
        parser.error("actor参数范围非法")
    if args.expected_on_field_players < 1 or args.team_clusters not in (2, 3):
        parser.error("应设置 expected_on_field_players>=1 且 team_clusters 为2或3")
    if not 0 <= args.min_turf_score <= 1 or not 0 <= args.min_track_turf_ratio <= 1:
        parser.error("球场过滤阈值必须在0..1")
    if not 0 <= args.min_foot_y_ratio <= 1:
        parser.error("min_foot_y_ratio必须在0..1")
    if not 0 <= args.min_geometry_ratio <= 1 or args.team_prototype_margin < 0:
        parser.error("min_geometry_ratio必须在0..1，team_prototype_margin必须>=0")
    args.keep_all_clusters = not args.drop_extra_clusters
    args.render_unassigned = not args.no_render_unassigned
    args.onboard_data = load_config(args.onboard_config) if args.onboard_config else {}
    args.field_geometry = get_value(args.onboard_data, "field_filter.geometry", {})
    args.team_prototypes = get_value(args.onboard_data, "identity.team_prototypes", [])
    return args


def export_ball_positions(path: Path, ball_pos: dict[int, tuple[float, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_proc", "ball_x_px", "ball_y_px", "observed"])
        for frame, point in sorted(ball_pos.items()):
            x, y = point[:2]
            writer.writerow([int(frame), round(float(x), 3), round(float(y), 3), 1])


def export_id_summary(path: Path, local_to_global: dict[int, int], tracklets: dict, total_frames: int) -> None:
    grouped: dict[int, dict] = {}
    for local_id, global_id in local_to_global.items():
        row = grouped.setdefault(int(global_id), {"global_id": int(global_id), "local_track_ids": [], "frames": set()})
        row["local_track_ids"].append(int(local_id))
        row["frames"].update(int(frame) for frame in tracklets[local_id]["frames"])
    result = []
    for global_id in sorted(grouped):
        row = grouped[global_id]
        frames = sorted(row.pop("frames"))
        row["local_track_ids"].sort()
        row["visible_frame_count"] = len(frames)
        row["presence_ratio"] = round(len(frames) / max(total_frames, 1), 6)
        row["first_frame_proc"] = frames[0] if frames else None
        row["last_frame_proc"] = frames[-1] if frames else None
        result.append(row)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def assert_unique_global_frames(local_to_global: dict[int, int], tracklets: dict) -> None:
    """Fail before export if global reassociation made one ID occupy two boxes."""
    owners: dict[tuple[int, int], int] = {}
    conflicts = []
    for local_id, global_id in local_to_global.items():
        for frame in tracklets[local_id]["frames"]:
            key = (int(frame), int(global_id))
            previous = owners.setdefault(key, int(local_id))
            if previous != int(local_id):
                conflicts.append({
                    "frame_proc": key[0], "global_id": key[1],
                    "local_track_ids": sorted({previous, int(local_id)}),
                })
    if conflicts:
        raise RuntimeError(
            "全局重关联违反同帧身份唯一性，拒绝导出 MOT: "
            + json.dumps(conflicts[:20], ensure_ascii=False)
        )


def enrich_events(
    outdir: Path, detections: list[tuple], ball_pos: dict[int, tuple[float, float]],
    local_to_global: dict[int, int], total_frames: int, fps: float, args: argparse.Namespace,
) -> list[dict]:
    source = outdir / "event_index.json"
    events = json.loads(source.read_text(encoding="utf-8"))
    boxes_by_frame = build_global_boxes(detections, local_to_global)
    bx, by, reliable = interpolate_ball(ball_pos, total_frames, args.ball_max_gap)
    for event in events:
        attribution = attribute_actor(
            int(event["event_frame_proc"]), fps, boxes_by_frame, bx, by, reliable,
            pre_seconds=args.actor_pre_sec, post_seconds=args.actor_post_sec,
            max_normalized_distance=args.actor_max_distance,
        )
        event.update(attribution)
        event["base_event_type"] = event.pop("event_type")
        event["model_main_dimension"] = None
        event["model_behavior_labels"] = []
        event["model_review_required"] = None
    enhanced_json = outdir / "event_index.json"
    enhanced_json.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = [
        "event_id", "event_time", "clip_start_time", "clip_end_time", "base_event_type",
        "score", "primary_global_id", "actor_attribution_status", "actor_attribution_reason",
        "event_frame_proc", "start_frame_proc", "end_frame_proc",
    ]
    with (outdir / "event_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)
    return events


def main() -> None:
    args = parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    cv2.setRNGSeed(0)
    core = load_tracking_core(args.tracking_core)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    args.outdir = str(outdir)
    args.weights = str(Path(args.weights))

    device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    if not torch.cuda.is_available():
        device, args.device = "cpu", "cpu"
    print(f"使用设备: {device}")
    reid = core.ReIDExtractor(device=device)
    detections, ball_pos, tracklets, total_frames = core.stage1_detect_track(args, reid)
    field_report = {"enabled": False}
    if not args.disable_field_filter:
        filtered = filter_tracklets_by_turf(
            args.video, detections, vid_stride=args.vid_stride,
            min_detection_score=args.min_turf_score,
            min_track_ratio=args.min_track_turf_ratio,
            min_foot_y_ratio=args.min_foot_y_ratio,
            field_geometry=args.field_geometry,
            min_geometry_ratio=args.min_geometry_ratio,
        )
        detections = filtered.detections
        restrict_tracklet_frames(tracklets, detections)
        field_report = {"enabled": True, **filtered.report}
        (outdir / "field_filter_report.json").write_text(
            json.dumps(field_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[Field] tracklet {field_report['input_tracklets']} -> "
            f"{field_report['kept_tracklets']}; detection "
            f"{field_report['input_detections']} -> {field_report['output_detections']}"
        )
    local_to_global = core.stage2_global_reassoc(tracklets, total_frames, args)
    assert_unique_global_frames(local_to_global, tracklets)

    meta = cv2.VideoCapture(args.video)
    raw_fps = meta.get(cv2.CAP_PROP_FPS) or 30.0
    meta.release()
    expected_fps = core.processed_fps(raw_fps, args.vid_stride)
    _, output_fps = core.stage3_render_and_mot(args, detections, local_to_global, tracklets)
    if not math.isclose(output_fps, expected_fps, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(f"处理帧率不一致: {output_fps} != {expected_fps}")
    core.stage4_events(
        args, detections, ball_pos, local_to_global, total_frames, raw_fps, output_fps
    )
    export_ball_positions(outdir / "ball_positions_observed.csv", ball_pos)
    export_id_summary(outdir / "global_id_summary.json", local_to_global, tracklets, total_frames)
    events = enrich_events(
        outdir, detections, ball_pos, local_to_global, total_frames, output_fps, args
    )
    run_meta = {
        "video": str(Path(args.video).resolve()), "raw_fps": raw_fps,
        "processed_fps": output_fps, "total_processed_frames": total_frames,
        "global_ids": len(set(local_to_global.values())), "event_candidates": len(events),
        "expected_on_field_players": args.expected_on_field_players,
        "team_clusters": args.team_clusters,
        "team_prototype_constraint": {
            "enabled": bool(args.team_prototypes),
            "prototype_count": len(args.team_prototypes),
            "confidence_margin": args.team_prototype_margin,
        },
        "identity_policy": "conservative_keep_all_clusters" if args.keep_all_clusters else "limited_top_n_cutoff",
        "field_filter": {k: v for k, v in field_report.items() if k != "tracks"},
        "edge_margin_seconds": args.edge_margin,
        "evaluation_policy": "eight_dimension_candidate_labels_plus_human_review",
        "multimodal_direct_scoring": "frozen_not_in_default_pipeline",
        "highlight_policy": {
            "candidate_events": args.n_clips,
            "source_seconds_before_event": args.pre_sec,
            "source_seconds_after_event": args.post_sec,
            "seconds_per_event": args.pre_sec + args.post_sec,
            "default_compilation_duration_seconds": args.n_clips * (args.pre_sec + args.post_sec),
        },
        "note": "global_id仅在本次运行内有效；低置信度事件必须人工复核",
    }
    (outdir / "tracking_run_metadata.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
