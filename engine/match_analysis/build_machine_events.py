from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2

from analysis_lib.tracking_adapter import (
    attribute_actor,
    build_global_boxes,
    interpolate_ball,
)
from analysis_lib.semantic_events import derive_semantic_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把现有 Stage4、tracking 球员归因和 match analysis 持球事件合成标准事件表"
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--ball", type=Path, required=True)
    parser.add_argument("--stage4-events", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage4-sample-size", type=int, default=20)
    parser.add_argument("--ball-max-gap", type=int, default=30)
    parser.add_argument("--field-length-m", type=float, default=45.0)
    parser.add_argument("--field-width-m", type=float, default=25.0)
    return parser.parse_args()


def _video_metadata(path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    if fps <= 0 or total <= 0:
        raise RuntimeError(f"视频元数据无效: fps={fps}, frames={total}")
    return fps, total


def _read_mot(path: Path) -> list[tuple]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, values in enumerate(csv.reader(handle), 1):
            if not values:
                continue
            if len(values) < 7:
                raise ValueError(f"MOT 第 {line_number} 行少于 7 列")
            frame = int(float(values[0])) - 1
            gid = int(float(values[1]))
            x, y, w, h, confidence = map(float, values[2:7])
            rows.append((frame, gid, x, y, w, h, confidence))
    if not rows:
        raise ValueError(f"MOT 为空: {path}")
    return rows


def _read_ball(path: Path) -> dict[int, tuple[float, float]]:
    result = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("observed", "1")).strip().lower() not in {"1", "true", "yes"}:
                continue
            result[int(row["frame_proc"])] = (
                float(row["ball_x_px"]), float(row["ball_y_px"]),
            )
    if not result:
        raise ValueError(f"足球观测为空: {path}")
    return result


def _stratified_stage4(rows: list[dict], limit: int) -> list[dict]:
    """Directly preserve Stage4's class-round-robin, within-class score ordering."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        # Raw Stage4 exports use ``event_type``.  The tracking delivery wrapper
        # enriches the same rows with actor attribution and renames that field
        # to ``base_event_type``.  Accept both contracts so the documented
        # tracking-output -> match analysis-input path works without rewriting artifacts.
        event_type = row.get("event_type", row.get("base_event_type"))
        if not event_type:
            raise KeyError("Stage4 event is missing event_type/base_event_type")
        groups[str(event_type)].append(row)
    for event_type in groups:
        groups[event_type].sort(key=lambda row: float(row["score"]), reverse=True)
    types = sorted(groups)
    selected: list[dict] = []
    type_index = 0
    target = min(max(0, limit), len(rows))
    while len(selected) < target and any(groups[event_type] for event_type in types):
        event_type = types[type_index % len(types)]
        type_index += 1
        if groups[event_type]:
            selected.append(groups[event_type].pop(0))
    return sorted(selected, key=lambda row: int(row["event_id"]))


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _window(anchor: float, duration: float, pre: float = 3.0, post: float = 2.0) -> tuple[float, float]:
    return round(max(0.0, anchor - pre), 3), round(min(duration, anchor + post), 3)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fps, total_frames = _video_metadata(args.video.resolve())
    duration = total_frames / fps

    mot_rows = _read_mot(args.mot.resolve())
    identity_map = {int(row[1]): int(row[1]) for row in mot_rows}
    boxes = build_global_boxes(mot_rows, identity_map)
    observed_ball = _read_ball(args.ball.resolve())
    ball_x, ball_y, reliable = interpolate_ball(
        observed_ball, total_frames, args.ball_max_gap,
    )

    raw_stage4 = json.loads(args.stage4_events.read_text(encoding="utf-8"))
    stage4_rows = raw_stage4["events"] if isinstance(raw_stage4, dict) else raw_stage4
    selected = _stratified_stage4(stage4_rows, args.stage4_sample_size)
    events: list[dict] = []
    for row in selected:
        frame = int(row["event_frame_proc"])
        actor = attribute_actor(frame, fps, boxes, ball_x, ball_y, reliable)
        start, end = _window(frame / fps, duration)
        candidates = actor["actor_candidates"]
        event_type = row.get("event_type", row.get("base_event_type"))
        events.append({
            "event_id": f"action_{int(row['event_id']):03d}",
            "start_time": start,
            "end_time": end,
            "primary_global_id": actor["primary_global_id"],
            "secondary_global_id": None,
            "event_type": str(event_type),
            "video_anchor_path": "",
            "jersey_number": None,
            "confidence": None,
            "source_kind": "stage4_action_candidate",
            "machine_status": "candidate_requires_review",
            "actor_assignment_status": actor["actor_attribution_status"],
            "actor_assignment_reason": actor["actor_attribution_reason"],
            "actor_candidates": candidates,
            "source_event_frame_proc": frame,
            "source_signal_score": float(row["score"]),
            "confidence_semantics": "event probability is not calibrated; actor ranking scores are kept only in actor_candidates",
            "description": "球运动/球员速度代理信号候选；类型与球员归属均需人工复核。",
        })

    transition_rows = _read_csv(args.analysis_dir / "possession_transitions.csv")
    for row in transition_rows:
        anchor = float(row["release_time_seconds"])
        start, end = _window(anchor, duration)
        events.append({
            "event_id": f"transition_{int(row['transition_id']):04d}",
            "start_time": start,
            "end_time": end,
            "primary_global_id": int(row["from_global_id"]),
            "secondary_global_id": int(row["to_global_id"]),
            "event_type": "possession_transition",
            "video_anchor_path": "",
            "jersey_number": None,
            "confidence": None,
            "source_kind": "stable_possession_transition",
            "machine_status": "candidate_requires_review",
            "classification": row["classification"],
            "from_team_id": row["from_team_id"],
            "to_team_id": row["to_team_id"],
            "release_time_seconds": float(row["release_time_seconds"]),
            "receive_time_seconds": float(row["receive_time_seconds"]),
            "displacement_m": float(row["displacement_m"]),
            "confidence_semantics": "no calibrated probability; backed by stable-possession geometry",
            "description": "稳定持球者 A→B 的变化；包括同队、对手、解围及潜在 ID 跳变。",
        })

    pass_rows = _read_csv(args.analysis_dir / "pass_events.csv")
    for row in pass_rows:
        anchor = float(row["release_time_seconds"])
        start, end = _window(anchor, duration)
        events.append({
            "event_id": f"pass_{int(row['pass_id']):04d}",
            "start_time": start,
            "end_time": end,
            "primary_global_id": int(row["from_global_id"]),
            "secondary_global_id": int(row["to_global_id"]),
            "event_type": "active_directed_pass_candidate",
            "video_anchor_path": "",
            "jersey_number": None,
            "confidence": None,
            "source_kind": "pass_network_candidate",
            "machine_status": "candidate_requires_review",
            "team_id": row["team_id"],
            "release_time_seconds": float(row["release_time_seconds"]),
            "receive_time_seconds": float(row["receive_time_seconds"]),
            "distance_m": float(row["distance_m"]),
            "direction_angle_deg": float(row["direction_angle_deg"]),
            "intent_proxy": row["intent_proxy"],
            "confidence_semantics": "no calibrated probability; same-team stable-control metric proxy",
            "description": "同队、稳定控球、具有米制方向位移的主动传递候选；需人工确认战术意图。",
        })

    running_rows = _read_csv(args.analysis_dir.parent / "metric_running" / "player_running_timeseries.csv")
    positions = {}
    for row in running_rows:
        try:
            positions[(int(row["proc_idx"]), int(row["global_id"]))] = (
                float(row.get("x_m_smooth") or row.get("x_m_raw")),
                float(row.get("y_m_smooth") or row.get("y_m_raw")),
            )
        except (KeyError, TypeError, ValueError):
            continue
    team_rows = _read_csv(args.analysis_dir / "player_team_map.csv")
    team_map = {int(row["global_id"]): str(row.get("team_id") or "") for row in team_rows}
    possession_rows = _read_csv(args.analysis_dir / "possession_intervals.csv")
    evidence_rows = _read_csv(args.analysis_dir / "possession_frame_evidence.csv")
    ball_metric = {}
    for row in evidence_rows:
        try:
            ball_metric[int(row["frame_proc"])] = (float(row["ball_x_m"]), float(row["ball_y_m"]))
        except (KeyError, TypeError, ValueError):
            continue
    semantic_events = derive_semantic_events(
        fps=fps,
        duration_seconds=duration,
        field_length_m=args.field_length_m,
        field_width_m=args.field_width_m,
        team_map=team_map,
        positions=positions,
        possessions=possession_rows,
        transitions=transition_rows,
        stage4_events=events[:len(selected)],
        ball_metric_by_frame=ball_metric,
    )
    events.extend(semantic_events)
    semantic_path = args.analysis_dir / "semantic_events.json"
    semantic_path.write_text(json.dumps({"schema_version": 1, "events": semantic_events}, ensure_ascii=False, indent=2), encoding="utf-8")

    events.sort(key=lambda row: (float(row["start_time"]), str(row["event_id"])))
    payload = {
        "schema_version": "events-for-annotation-v1",
        "data_status": "machine_candidates_without_human_labels",
        "total_events": len(events),
        "counts": {
            "stage4_action_candidates": len(selected),
            "possession_transitions": len(transition_rows),
            "active_directed_pass_candidates": len(pass_rows),
            "semantic_event_candidates": len(semantic_events),
        },
        "events": events,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["counts"], "total": len(events)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
