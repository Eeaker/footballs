from __future__ import annotations

import math
from typing import Any


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _window(frame: int, fps: float, duration: float, pre: float = 3.0, post: float = 2.0) -> tuple[float, float]:
    anchor = frame / fps
    return round(max(0.0, anchor - pre), 3), round(min(duration, anchor + post), 3)


def derive_semantic_events(
    *,
    fps: float,
    duration_seconds: float,
    field_length_m: float,
    field_width_m: float,
    team_map: dict[int, str],
    positions: dict[tuple[int, int], tuple[float, float]],
    possessions: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    stage4_events: list[dict[str, Any]],
    ball_metric_by_frame: dict[int, tuple[float, float]],
    pressure_distance_m: float = 1.5,
    counterpress_seconds: float = 5.0,
) -> list[dict[str, Any]]:
    """Create explainable candidates from tracking and stable-possession evidence.

    These are deliberately candidates: single-camera geometry cannot prove the
    football meaning without a reviewer. Every row keeps the rule and evidence.
    """
    fps = max(1.0, float(fps))
    result: list[dict[str, Any]] = []
    positions_by_frame: dict[int, list[tuple[int, tuple[float, float]]]] = {}
    for (frame, gid), point in positions.items():
        positions_by_frame.setdefault(frame, []).append((gid, point))

    # Retained possession while an opponent is within pressure distance.
    min_frames = max(3, int(round(0.6 * fps)))
    for index, row in enumerate(possessions):
        gid = _i(row.get("global_id"), -1)
        team_id = str(row.get("team_id") or team_map.get(gid, ""))
        start = _i(row.get("start_frame_proc"), -1)
        end = _i(row.get("end_frame_proc"), -1)
        if gid < 0 or start < 0 or end - start + 1 < min_frames:
            continue
        pressured: list[tuple[int, float, int]] = []
        for frame in range(start, end + 1):
            own = positions.get((frame, gid))
            if own is None:
                continue
            nearest: tuple[float, int] | None = None
            for other_gid, other in positions_by_frame.get(frame, []):
                if other_gid == gid or team_map.get(other_gid) == team_id:
                    continue
                distance = math.dist(own, other)
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, other_gid)
            if nearest and nearest[0] <= pressure_distance_m:
                pressured.append((frame, nearest[0], nearest[1]))
        if len(pressured) < min_frames:
            continue
        anchor, distance, opponent = min(pressured, key=lambda value: value[1])
        clip_start, clip_end = _window(anchor, fps, duration_seconds)
        result.append({
            "event_id": f"shield_{index:04d}",
            "event_type": "shielding_under_pressure",
            "label": "对抗护球",
            "primary_global_id": gid,
            "secondary_global_id": opponent,
            "team_id": team_id,
            "event_frame_proc": anchor,
            "start_time": clip_start,
            "end_time": clip_end,
            "machine_status": "candidate_requires_review",
            "confidence": None,
            "evidence": {"pressured_frames": len(pressured), "minimum_opponent_distance_m": round(distance, 3), "retained_until_frame": end},
            "description": "近身对手施压期间仍保持稳定球权的候选片段。",
        })

    # A team loses the ball, then wins it back within five seconds.
    opponent_changes = [row for row in transitions if str(row.get("classification")) == "opponent_possession_change"]
    opponent_changes.sort(key=lambda row: _i(row.get("receive_frame_proc"), _i(row.get("release_frame_proc"))))
    max_gap = int(round(counterpress_seconds * fps))
    for index, lost in enumerate(opponent_changes):
        lost_team = str(lost.get("from_team_id") or "")
        lost_frame = _i(lost.get("receive_frame_proc"), _i(lost.get("release_frame_proc")))
        for recovered in opponent_changes[index + 1:]:
            recovery_frame = _i(recovered.get("receive_frame_proc"), _i(recovered.get("release_frame_proc")))
            if recovery_frame - lost_frame > max_gap:
                break
            if str(recovered.get("to_team_id") or "") != lost_team:
                continue
            recovery_gid = _i(recovered.get("to_global_id"), -1)
            if recovery_gid < 0:
                break
            clip_start, clip_end = _window(recovery_frame, fps, duration_seconds, 5.0, 2.0)
            result.append({
                "event_id": f"counterpress_{index:04d}_{recovery_frame}",
                "event_type": "counterpress_recovery",
                "label": "丢球反抢",
                "primary_global_id": recovery_gid,
                "secondary_global_id": _i(lost.get("from_global_id"), -1),
                "team_id": lost_team,
                "event_frame_proc": recovery_frame,
                "start_time": clip_start,
                "end_time": clip_end,
                "window_seconds": float(counterpress_seconds),
                "machine_status": "candidate_requires_review",
                "confidence": None,
                "evidence": {"lost_frame": lost_frame, "recovered_frame": recovery_frame, "recovery_delay_seconds": round((recovery_frame - lost_frame) / fps, 3)},
                "description": "球队丢失球权后 5 秒内重新获得稳定球权的候选片段。",
            })
            break

    # A shot candidate followed by the ball entering a goal-line mouth zone.
    goal_half_width = max(1.5, min(3.66, field_width_m * 0.12))
    goal_y_min, goal_y_max = field_width_m / 2 - goal_half_width, field_width_m / 2 + goal_half_width
    goal_margin = max(0.6, field_length_m * 0.025)
    for row in stage4_events:
        event_type = str(row.get("event_type") or row.get("base_event_type") or "").lower()
        if "shot" not in event_type and "射门" not in event_type:
            continue
        shot_frame = _i(row.get("event_frame_proc"), _i(row.get("source_event_frame_proc"), -1))
        if shot_frame < 0:
            continue
        goal_frame = None
        goal_point = None
        for frame in range(shot_frame, shot_frame + int(round(2.0 * fps)) + 1):
            point = ball_metric_by_frame.get(frame)
            if point and (point[0] <= goal_margin or point[0] >= field_length_m - goal_margin) and goal_y_min <= point[1] <= goal_y_max:
                goal_frame, goal_point = frame, point
                break
        if goal_frame is None:
            continue
        clip_start, clip_end = _window(goal_frame, fps, duration_seconds, 4.0, 3.0)
        result.append({
            "event_id": f"goal_{_i(row.get('event_id'), shot_frame):04d}",
            "event_type": "goal_candidate",
            "label": "进球候选",
            "primary_global_id": _i(row.get("primary_global_id"), -1),
            "secondary_global_id": None,
            "team_id": row.get("team_id"),
            "event_frame_proc": goal_frame,
            "start_time": clip_start,
            "end_time": clip_end,
            "machine_status": "candidate_requires_review",
            "confidence": None,
            "evidence": {"shot_frame": shot_frame, "goal_zone_frame": goal_frame, "ball_x_m": round(goal_point[0], 3), "ball_y_m": round(goal_point[1], 3)},
            "description": "射门候选后足球进入球门线门幅区域；需人工确认是否完整越线且未被遮挡。",
        })

    return sorted(result, key=lambda row: (row["event_frame_proc"], row["event_id"]))
