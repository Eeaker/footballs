"""Build transferable player-card packages from verified identities and metric tracks."""

from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import yaml


SPRINT_THRESHOLD_MPS = 4.5
SPRINT_MIN_DURATION_SEC = 0.5
SPRINT_MAX_GAP_SEC = 0.125
RELIABLE_PEAK_WINDOW_SEC = 0.5
ABNORMAL_SPEED_MPS = 15.0


ASSESSMENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "assessment_report_input.schema.json",
    "title": "NATI player assessment report input",
    "type": "object",
    "required": [
        "schema_version", "data_status", "player", "match", "headline",
        "key_metrics", "radar", "overall", "analysis", "style_archetype",
        "position_recommendations", "messages", "evidence",
    ],
    "properties": {
        "schema_version": {"const": "nati-assessment-input-v1"},
        "data_status": {"enum": ["mock", "partial", "evaluation_pending", "final"]},
        "player": {"type": "object"},
        "match": {"type": "object"},
        "headline": {"type": "object"},
        "key_metrics": {"type": "array", "maxItems": 3},
        "radar": {"type": "object"},
        "overall": {"type": "object"},
        "analysis": {"type": "object"},
        "style_archetype": {"type": "object"},
        "position_recommendations": {"type": "array", "maxItems": 3},
        "messages": {"type": "object"},
        "evidence": {"type": "object"},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_confirmed_players(path: str | Path) -> tuple[dict[str, dict], dict]:
    """Load the verifier buckets; conflicts/mismatches never enter a player card."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    players: dict[str, dict] = {}
    for item in data.get("eligible_confirmed", []):
        team = str(item.get("team") or "").strip().lower()
        number = str(item.get("final_number") or "").strip()
        if team not in {"white", "yellow", "blue"} or not number:
            continue
        player_id = f"{team}_{number}"
        player = players.setdefault(player_id, {
            "player_id": player_id, "team": team, "jersey_number": int(number),
            "status": "eligible_confirmed", "global_ids": [], "confidence": [],
            "global_id_confidence": {},
        })
        gid = int(item["global_id"])
        confidence = float(item.get("confidence") or 0.0)
        player["global_ids"].append(gid)
        player["confidence"].append(confidence)
        player["global_id_confidence"][str(gid)] = round(confidence, 6)
    for player in players.values():
        player["global_ids"].sort()
        player["identity_confidence"] = round(min(player.pop("confidence"), default=0.0), 6)
    audit = {
        "eligible_confirmed_global_ids": sum(len(p["global_ids"]) for p in players.values()),
        "confirmed_player_identities": len(players),
        "excluded_conflict": len(data.get("excluded_conflict", [])),
        "excluded_mismatch": len(data.get("excluded_mismatch", [])),
        "excluded_unreadable": len(data.get("excluded_unreadable", [])),
        "policy": "eligible_confirmed only; aliases with same team+number are aggregated",
    }
    return players, audit


def group_timeseries(path: str | Path) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in read_csv(path):
        grouped[int(row["global_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["proc_idx"]))
    return dict(grouped)


def _float(row: dict, key: str) -> float | None:
    value = row.get(key)
    return None if value in (None, "") else float(value)


def resolve_player_metric_ids(player: dict, timeseries: dict[int, list[dict]], fps: float) -> dict:
    """Prevent simultaneously visible, spatially separate IDs from being summed.

    Multiple IDs with the same OCR number may be true temporal fragments, but
    they cannot be the same person when they coexist far apart on the pitch.
    In that case metrics use the highest-confidence ID and the conflict remains
    explicit in the package for review.
    """
    gids = list(player["global_ids"])
    confidence = player.get("global_id_confidence", {})
    conflicts = []
    coordinates: dict[int, dict[int, tuple[float, float]]] = {}
    for gid in gids:
        coordinates[gid] = {
            int(row["proc_idx"]): (float(row["x_m_smooth"]), float(row["y_m_smooth"]))
            for row in timeseries.get(gid, [])
            if row.get("x_m_smooth") not in (None, "") and row.get("y_m_smooth") not in (None, "")
        }
    minimum_overlap = max(3, int(round(0.5 * fps)))
    for left_index, left in enumerate(gids):
        for right in gids[left_index + 1:]:
            overlap = sorted(set(coordinates[left]) & set(coordinates[right]))
            if len(overlap) < minimum_overlap:
                continue
            distances = [
                float(np.hypot(coordinates[left][frame][0] - coordinates[right][frame][0],
                               coordinates[left][frame][1] - coordinates[right][frame][1]))
                for frame in overlap
            ]
            median_distance = float(np.median(distances))
            if median_distance > 2.0:
                conflicts.append({
                    "global_ids": [left, right], "overlap_frames": len(overlap),
                    "overlap_sec": round(len(overlap) / fps, 3),
                    "median_separation_m": round(median_distance, 3),
                })
    if conflicts:
        canonical = max(gids, key=lambda gid: (float(confidence.get(str(gid), 0.0)), -gid))
        metric_gids = [canonical]
        status = "identity_overlap_conflict_canonical_only"
    else:
        metric_gids = gids
        status = "verified_aliases_aggregated" if len(gids) > 1 else "single_verified_id"
    return {
        "status": status, "metric_global_ids": metric_gids,
        "excluded_metric_global_ids": [gid for gid in gids if gid not in metric_gids],
        "overlap_conflicts": conflicts,
        "rule": "spatially separate IDs overlapping >=0.5 s are not aggregated; highest OCR-confidence ID is used",
    }


def calculate_player_running(rows: list[dict], fps: float) -> tuple[dict, list[dict]]:
    valid_speeds: list[float] = []
    total_distance = high_speed_distance = 0.0
    abnormal_rows = 0
    valid_frames: set[int] = set()
    heatmap_rows: list[dict] = []
    grouped: dict[tuple[int, int], list[tuple[int, float | None, float | None]]] = defaultdict(list)
    for row in sorted(rows, key=lambda value: (int(value["global_id"]), int(value["proc_idx"]))):
        gid, frame = int(row["global_id"]), int(row["proc_idx"])
        segment = int(row.get("segment_id") or 0)
        speed = _float(row, "speed_mps")
        step = _float(row, "step_distance_m")
        if speed is not None and speed > ABNORMAL_SPEED_MPS:
            abnormal_rows += 1
            continue
        valid_frames.add(frame)
        heatmap_rows.append(row)
        if speed is not None:
            valid_speeds.append(speed)
        if step is not None:
            total_distance += step
        grouped[(gid, segment)].append((frame, speed, step))

    min_sprint_frames = max(1, int(np.ceil(SPRINT_MIN_DURATION_SEC * fps)))
    max_gap_frames = max(0, int(round(SPRINT_MAX_GAP_SEC * fps)))
    peak_window_frames = max(1, int(np.ceil(RELIABLE_PEAK_WINDOW_SEC * fps)))
    sprint_count = 0
    reliable_peaks: list[float] = []

    for records in grouped.values():
        # Segment IDs can still contain missing frames. Split them so a tracking
        # gap can never manufacture a sprint or a peak-speed window.
        sequences: list[list[tuple[int, float | None, float | None]]] = []
        for record in sorted(records):
            if not sequences or record[0] != sequences[-1][-1][0] + 1:
                sequences.append([])
            sequences[-1].append(record)
        for sequence in sequences:
            numeric_runs: list[list[tuple[int, float, float | None]]] = []
            for frame, speed, step in sequence:
                if speed is None:
                    continue
                if not numeric_runs or frame != numeric_runs[-1][-1][0] + 1:
                    numeric_runs.append([])
                numeric_runs[-1].append((frame, speed, step))
            for run in numeric_runs:
                speeds = [item[1] for item in run]
                if len(speeds) >= peak_window_frames:
                    reliable_peaks.extend(
                        float(np.median(speeds[start:start + peak_window_frames]))
                        for start in range(len(speeds) - peak_window_frames + 1)
                    )

                high_indices = [index for index, speed in enumerate(speeds)
                                if speed > SPRINT_THRESHOLD_MPS]
                if not high_indices:
                    continue
                candidates: list[list[int]] = [[high_indices[0]]]
                for index in high_indices[1:]:
                    if index - candidates[-1][-1] - 1 <= max_gap_frames:
                        candidates[-1].append(index)
                    else:
                        candidates.append([index])
                for candidate in candidates:
                    # Count only frames actually above threshold. A short dip may
                    # join a bout, but cannot satisfy the duration requirement.
                    if len(candidate) < min_sprint_frames:
                        continue
                    sprint_count += 1
                    for index in range(candidate[0], candidate[-1] + 1):
                        step = run[index][2]
                        if step is not None:
                            high_speed_distance += step

    max_speed = max(reliable_peaks, default=0.0)
    speed_p95 = float(np.percentile(valid_speeds, 95)) if valid_speeds else 0.0
    summary = {
        "total_distance_m": round(total_distance, 3),
        "sprint_count": sprint_count,
        "max_speed_mps": round(max_speed, 3),
        "speed_p95_mps": round(speed_p95, 3),
        "high_speed_distance_m": round(high_speed_distance, 3),
        "tracked_visible_time_sec": round(len(valid_frames) / fps, 3),
        "playing_time_sec": None,
        "valid_frame_count": len(valid_frames),
        "abnormal_speed_rows_excluded": abnormal_rows,
        "sprint_min_duration_sec": SPRINT_MIN_DURATION_SEC,
        "sprint_definition": (
            f"speed > {SPRINT_THRESHOLD_MPS} m/s for at least {SPRINT_MIN_DURATION_SEC} s; "
            f"gaps <= {SPRINT_MAX_GAP_SEC} s may be joined"
        ),
        "max_speed_definition": (
            f"maximum rolling median speed over {RELIABLE_PEAK_WINDOW_SEC} s of contiguous tracking"
        ),
        "time_definition": "tracked_visible_time_sec is detection-visible time, not roster playing time",
        "speed_quality_rule": f"speed > {ABNORMAL_SPEED_MPS} m/s excluded from all card metrics",
    }
    return summary, heatmap_rows


def render_heatmap(rows: list[dict], bounds: dict, output: str | Path) -> dict:
    output = Path(output)
    width, height = 600, 400
    canvas = np.full((height, width, 3), (50, 125, 46), dtype=np.uint8)
    xmin, xmax = float(bounds["x_min"]), float(bounds["x_max"])
    ymin, ymax = float(bounds["y_min"]), float(bounds["y_max"])
    margin = 24
    pitch_w = width - 2 * margin
    pitch_h = int(round(pitch_w * (ymax - ymin) / max(xmax - xmin, 1e-9)))
    pitch_h = min(pitch_h, height - 2 * margin)
    x0, y0 = (width - pitch_w) // 2, (height - pitch_h) // 2
    xs = [float(row["x_m_smooth"]) for row in rows]
    ys = [float(row["y_m_smooth"]) for row in rows]
    bins_x, bins_y = 100, max(10, int(round(100 * (ymax - ymin) / (xmax - xmin))))
    if xs:
        hist, _, _ = np.histogram2d(xs, ys, bins=(bins_x, bins_y), range=((xmin, xmax), (ymin, ymax)))
        hist = cv2.GaussianBlur(hist.astype(np.float32), (0, 0), 2.2)
        peak = float(hist.max())
        normalized = np.zeros_like(hist, dtype=np.uint8) if peak <= 0 else np.uint8(np.clip(hist / peak * 255, 0, 255))
        density = cv2.resize(normalized.T, (pitch_w, pitch_h), interpolation=cv2.INTER_CUBIC)
        color = cv2.applyColorMap(density, cv2.COLORMAP_JET)
        alpha = (density.astype(np.float32) / 255.0 * .72)[..., None]
        roi = canvas[y0:y0 + pitch_h, x0:x0 + pitch_w].astype(np.float32)
        canvas[y0:y0 + pitch_h, x0:x0 + pitch_w] = np.uint8(roi * (1 - alpha) + color * alpha)
    white = (245, 245, 245)
    cv2.rectangle(canvas, (x0, y0), (x0 + pitch_w, y0 + pitch_h), white, 2)
    centre = (x0 + pitch_w // 2, y0 + pitch_h // 2)
    cv2.line(canvas, (centre[0], y0), (centre[0], y0 + pitch_h), white, 2)
    radius = max(4, int(round(3.0 / (xmax - xmin) * pitch_w)))
    cv2.circle(canvas, centre, radius, white, 2)
    box_w, box_h = int(pitch_w * .13), int(pitch_h * .45)
    cv2.rectangle(canvas, (x0, centre[1] - box_h // 2), (x0 + box_w, centre[1] + box_h // 2), white, 2)
    cv2.rectangle(canvas, (x0 + pitch_w - box_w, centre[1] - box_h // 2),
                  (x0 + pitch_w, centre[1] + box_h // 2), white, 2)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"cannot write heatmap: {output}")
    return {"file_path": output.name, "pitch_size_m": [xmax - xmin, ymax - ymin],
            "resolution": [bins_x, bins_y], "image_size_px": [width, height], "samples": len(rows)}


def assessment_interface(player: dict, running: dict) -> dict:
    """Interface only: downstream evaluators fill semantic scores and prose."""
    return {
        "schema_version": "nati-assessment-input-v1",
        "data_status": "evaluation_pending",
        "player": {
            "player_id": player["player_id"], "team": player["team"],
            "jersey_number": player["jersey_number"], "age_group": "U12",
            "position": None, "preferred_foot": None,
        },
        "match": {"report_id": None, "assessment_date": None, "tournament": None, "season": None,
                  "match_accolade": None},
        "headline": {"nickname": None, "quote": None},
        "key_metrics": [
            {"key": "total_distance_m", "value": running["total_distance_m"], "label": "跑动距离", "source": "running.json"},
            {"key": "sprint_count", "value": running["sprint_count"], "label": "冲刺次数", "source": "running.json"},
            {"key": "max_speed_mps", "value": running["max_speed_mps"], "label": "可靠峰值速度", "source": "running.json"},
        ],
        "radar": {"dimensions": {key: None for key in ["体", "技", "战", "心", "智", "观", "决", "位"]},
                  "scale": [0, 100], "source": "semantic_labels_pending"},
        "overall": {"ca_score": None, "potential_grade": None, "potential_direction": None},
        "analysis": {"strengths": [], "improvements": []},
        "style_archetype": {
            "triangle": {"tactical_literacy": None, "physical_competition": None, "talent": None},
            "reference_player": None, "similarities": [], "differences": [], "next_target": None,
        },
        "position_recommendations": [],
        "messages": {"to_player": None, "to_family_and_coach": None},
        "evidence": {"running": "running.json", "heatmap": "heatmap.png",
                     "events": "events_for_annotation.json", "highlights_dir": "highlights/"},
    }


def _copy_event_clip(event: dict, destination: Path, *, video: Path, events_base: Path) -> Path:
    source_value = str(event.get("video_anchor_path") or "").strip()
    source = Path(source_value) if source_value else None
    if source is not None and not source.is_absolute():
        source = events_base / source
    if source is not None and source.is_file():
        shutil.copy2(source, destination)
        return destination

    start = event.get("start_time")
    end = event.get("end_time")
    if start in (None, "") or end in (None, ""):
        raise FileNotFoundError(
            f"事件既没有有效切片，也没有 start_time/end_time: {event.get('event_id')}"
        )
    start_sec, end_sec = max(0.0, float(start)), float(end)
    if end_sec <= start_sec:
        raise ValueError(f"事件时间范围无效: {start_sec} - {end_sec}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开源视频裁剪事件: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    start_frame = max(0, min(total, int(np.floor(start_sec * fps))))
    end_frame = max(start_frame + 1, min(total, int(np.ceil(end_sec * fps))))
    destination = destination.with_suffix(".mp4")
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建事件切片: {destination}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    for _ in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    if written == 0:
        raise RuntimeError(f"事件切片没有写出任何帧: {event.get('event_id')}")
    return destination


def load_mot_boxes(path: str | Path) -> dict[int, dict[int, tuple[float, float, float, float]]]:
    boxes: dict[int, dict[int, tuple[float, float, float, float]]] = defaultdict(dict)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for values in csv.reader(handle):
            if len(values) < 6:
                continue
            frame, gid = int(float(values[0])), int(float(values[1]))
            boxes[gid][frame] = tuple(float(value) for value in values[2:6])
    return dict(boxes)


def _box_for_frame(
    boxes: dict[int, tuple[float, float, float, float]], frame: int, max_gap_frames: int,
) -> tuple[float, float, float, float] | None:
    if frame in boxes:
        return boxes[frame]
    previous = next((candidate for candidate in range(frame - 1, frame - max_gap_frames - 1, -1)
                     if candidate in boxes), None)
    following = next((candidate for candidate in range(frame + 1, frame + max_gap_frames + 1)
                      if candidate in boxes), None)
    if previous is not None and following is not None:
        ratio = (frame - previous) / (following - previous)
        return tuple(a + ratio * (b - a) for a, b in zip(boxes[previous], boxes[following]))
    nearest = previous if previous is not None else following
    return boxes.get(nearest) if nearest is not None else None


def render_marked_event_clip(
    *, video: Path, destination: Path, start_sec: float, end_sec: float,
    target_gid: int, target_boxes: dict[int, tuple[float, float, float, float]],
    team: str, jersey_number: int | None,
) -> dict:
    """Render a stable target marker; interpolate only short tracking gaps."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开源视频标记高光: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    start_frame = max(0, min(total, int(np.floor(start_sec * fps))))
    end_frame = max(start_frame + 1, min(total, int(np.ceil(end_sec * fps))))
    max_gap_frames = max(1, int(round(0.5 * fps)))
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建标记高光: {destination}")
    team_colours = {"yellow": (30, 220, 255), "blue": (255, 120, 30), "white": (255, 255, 255)}
    color = team_colours.get(team, (255, 255, 255))
    label = (f"TARGET {team.upper()} #{jersey_number}" if jersey_number is not None
             else f"TARGET {team.upper()} UNKNOWN ID {target_gid}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    marked_frames = written = 0
    for source_frame in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        # MOT is one-based while OpenCV source_frame is zero-based.
        box = _box_for_frame(target_boxes, source_frame + 1, max_gap_frames)
        if box is not None:
            x, y, w, h = box
            x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
            x2, y2 = min(width - 1, int(round(x + w))), min(height - 1, int(round(y + h)))
            if x2 > x1 and y2 > y1:
                thickness = max(3, int(round(width / 640)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                centre_x = (x1 + x2) // 2
                arrow_tip = max(5, y1 - 4)
                arrow_top = max(0, arrow_tip - max(18, height // 32))
                cv2.arrowedLine(frame, (centre_x, arrow_top), (centre_x, arrow_tip),
                                color, thickness, tipLength=0.35)
                font_scale = max(0.55, width / 1800)
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                       font_scale, 2)
                label_y = min(height - 4, max(text_h + 6, y2 + text_h + 10))
                cv2.rectangle(frame, (x1, label_y - text_h - 6),
                              (min(width - 1, x1 + text_w + 8), label_y + 3), (25, 25, 25), -1)
                cv2.putText(frame, label, (x1 + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, color, 2, cv2.LINE_AA)
                marked_frames += 1
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    if written == 0:
        raise RuntimeError(f"标记高光没有写出任何帧: {destination}")
    probe = cv2.VideoCapture(str(destination))
    encoded_frames = int(round(probe.get(cv2.CAP_PROP_FRAME_COUNT))) if probe.isOpened() else 0
    probe.release()
    minimum_frames = max(1, int(written * 0.9))
    if not destination.is_file() or destination.stat().st_size <= 1024 or encoded_frames < minimum_frames:
        raise RuntimeError(
            f"标记高光写出不完整: {destination}（期望约 {written} 帧，实际 {encoded_frames} 帧）"
        )
    return {
        "enabled": True, "target_global_id": target_gid,
        "style": "bbox+arrow+label", "marked_frame_count": marked_frames,
        "total_frame_count": written, "short_gap_interpolation_sec": 0.5,
    }


def generate_player_card_data(
    *, video: str | Path, numbers: str | Path, events: str | Path,
    running_timeseries: str | Path, calibration: str | Path,
    output: str | Path, fps: float, source_mot: str | Path | None = None,
    running_quality: str | Path | None = None,
) -> dict:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output must not exist: {output}")
    work = output.with_name(output.name + f".work-{int(time.time())}")
    if work.exists():
        raise FileExistsError(work)
    work.mkdir(parents=True)
    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video missing: {video}")
    events_path = Path(events).resolve()
    timeseries_path = Path(running_timeseries).resolve()
    calibration_path = Path(calibration).resolve()
    players, identity_audit = load_confirmed_players(numbers)
    timeseries = group_timeseries(timeseries_path)
    mot_boxes = load_mot_boxes(source_mot) if source_mot else {}
    calibration_data = json.loads(calibration_path.read_text(encoding="utf-8"))
    bounds = calibration_data["field_bounds_m"]
    event_data = json.loads(events_path.read_text(encoding="utf-8"))
    all_events = event_data["events"] if isinstance(event_data, dict) else event_data
    gid_to_player = {gid: player_id for player_id, player in players.items() for gid in player["global_ids"]}
    generated_at = datetime.now(timezone.utc).isoformat()

    root_highlights = work / "highlights"
    root_highlights.mkdir()
    normalized_events = []
    for index, event in enumerate(all_events, 1):
        row = dict(event)
        event_id = str(row.get("event_id") or f"ev{index:03d}")
        row["event_id"] = event_id
        primary = row.get("primary_global_id")
        row["player_id"] = gid_to_player.get(int(primary)) if primary not in (None, "") else None
        # Only confirmed players consume the internal root clip.  Unresolved
        # players are exported later directly from source timestamps with a
        # target marker, so materialising every event here duplicates several
        # gigabytes without changing the formal package.
        if row["player_id"] is not None:
            source_suffix = Path(str(row.get("video_anchor_path") or "clip.mp4")).suffix or ".mp4"
            root_name = f"{event_id}{source_suffix}"
            root_clip = _copy_event_clip(
                row, root_highlights / root_name, video=video, events_base=events_path.parent,
            )
            row["video_anchor_path"] = f"highlights/{root_clip.name}"
        else:
            row["video_anchor_path"] = ""
        normalized_events.append(row)
    (work / "events_for_annotation.json").write_text(json.dumps({
        "schema_version": "events-for-annotation-v1", "total_events": len(normalized_events),
        "events": normalized_events,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows = []
    running_index = []
    for player_id, player in sorted(players.items()):
        player_dir = work / player_id
        highlights_dir = player_dir / "highlights"
        highlights_dir.mkdir(parents=True)
        identity_resolution = resolve_player_metric_ids(player, timeseries, fps)
        rows = [row for gid in identity_resolution["metric_global_ids"] for row in timeseries.get(gid, [])]
        running, heatmap_rows = calculate_player_running(rows, fps)
        heatmap = render_heatmap(heatmap_rows, bounds, player_dir / "heatmap.png")
        video_metadata = calibration_data["video_metadata"]
        duration_seconds = float(video_metadata.get("duration_seconds") or 0)
        if duration_seconds <= 0:
            total_frames = float(video_metadata.get("raw_total_frames") or video_metadata.get("proc_total_frames") or 0)
            source_fps = float(video_metadata.get("raw_fps") or video_metadata.get("proc_fps") or fps)
            duration_seconds = total_frames / source_fps if source_fps > 0 else 0
        player_with_resolution = dict(player)
        player_with_resolution["identity_resolution"] = identity_resolution
        identity = {
            "player": player_with_resolution,
            "source": {"video_file": video.name,
                       "video_duration_sec": round(duration_seconds, 3),
                       "processed_frames": int(video_metadata["proc_total_frames"])},
            "metadata": {"generated_at": generated_at, "generated_by": "generate_player_card.py"},
        }
        (player_dir / "identity.yaml").write_text(
            yaml.safe_dump(identity, allow_unicode=True, sort_keys=False), encoding="utf-8",
        )
        running_json = {
            "schema_version": "player-running-v2", "data_status": "measured_with_quality_filters",
            "player_id": player_id, "source_global_ids": player["global_ids"],
            "metric_global_ids": identity_resolution["metric_global_ids"],
            "identity_resolution": identity_resolution,
            "summary": running, "heatmap": heatmap,
            "raw_data": {"timeseries_file": "player_running_timeseries.csv",
                         "calibration_file": calibration_path.name,
                         "mot_file": Path(source_mot).name if source_mot else None},
        }
        (player_dir / "running.json").write_text(json.dumps(running_json, ensure_ascii=False, indent=2), encoding="utf-8")
        player_events = []
        for sequence, event in enumerate([row for row in normalized_events if row.get("player_id") == player_id], 1):
            suffix = Path(event["video_anchor_path"]).suffix
            event_number = f"{sequence:03d}"
            filename = f"{player_id}_ev{event_number}{suffix}"
            primary_gid = int(event["primary_global_id"])
            if source_mot and primary_gid in mot_boxes:
                marker = render_marked_event_clip(
                    video=video, destination=highlights_dir / filename,
                    start_sec=float(event["start_time"]), end_sec=float(event["end_time"]),
                    target_gid=primary_gid, target_boxes=mot_boxes[primary_gid],
                    team=player["team"], jersey_number=player["jersey_number"],
                )
            else:
                shutil.copy2(work / event["video_anchor_path"], highlights_dir / filename)
                marker = {"enabled": False, "reason": "source MOT or target track unavailable"}
            metadata = {
                "event_id": event["event_id"], "file_name": filename, "player_id": player_id,
                "time_range": {"start_sec": event["start_time"], "end_sec": event["end_time"],
                               "duration_sec": round(float(event["end_time"]) - float(event["start_time"]), 3)},
                "event_type": event["event_type"], "source_video": video.name,
                "confidence": event.get("confidence"), "identity_status": "eligible_confirmed",
                "marker": marker,
            }
            (highlights_dir / f"{Path(filename).stem}.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            player_event = dict(event)
            player_event["sequence"] = sequence
            player_event["source_video_anchor_path"] = player_event["video_anchor_path"]
            player_event["video_anchor_path"] = f"highlights/{filename}"
            player_event["video_file"] = player_event["video_anchor_path"]
            player_event["status"] = "pending_semantic_label"
            player_event["semantic_label"] = None
            player_events.append(player_event)
        annotation = {
            "schema_version": "events-for-annotation-v1", "player_id": player_id,
            "player_identity": {"team": player["team"], "jersey_number": player["jersey_number"]},
            "total_events": len(player_events), "events": player_events,
            "annotation_guide": {
                "dimensions": ["体", "技", "观", "决", "战", "心", "智", "位"],
                "instruction": "逐个播放相对路径视频，在Excel台账中填写八维评分和文字观察。算法事件类型仅供参考。",
                "output_file": f"{player_id}_semantic_labels.xlsx",
            },
        }
        (player_dir / "events_for_annotation.json").write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        (player_dir / "assessment_report_input.json").write_text(
            json.dumps(assessment_interface(player, running), ensure_ascii=False, indent=2), encoding="utf-8",
        )
        summary_rows.append({
            "player_id": player_id, "jersey_number": player["jersey_number"], "team": player["team"],
            "total_distance": running["total_distance_m"], "sprint_count": running["sprint_count"],
            "max_speed_mps": running["max_speed_mps"],
            "speed_p95_mps": running["speed_p95_mps"],
            "tracked_visible_time_sec": running["tracked_visible_time_sec"],
            "metric_global_ids": ";".join(str(gid) for gid in identity_resolution["metric_global_ids"]),
            "identity_resolution_status": identity_resolution["status"],
            "heatmap_data_path": f"{player_id}/heatmap.png",
            "data_quality": "measured_with_quality_filters",
        })
        running_index.append({"player_id": player_id, "running_path": f"{player_id}/running.json",
                              "heatmap_path": f"{player_id}/heatmap.png"})

    write_csv(work / "player_running_summary.csv", summary_rows, [
        "player_id", "jersey_number", "team", "total_distance", "sprint_count",
        "max_speed_mps", "speed_p95_mps", "tracked_visible_time_sec", "metric_global_ids",
        "identity_resolution_status", "heatmap_data_path", "data_quality",
    ])
    (work / "running.json").write_text(json.dumps({
        "schema_version": "player-running-index-v1", "players": running_index,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (work / "identity.yaml").write_text(yaml.safe_dump({
        "schema_version": "verified-identity-map-v1", "players": list(players.values()),
        "audit": identity_audit,
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (work / "assessment_report_input.schema.json").write_text(
        json.dumps(ASSESSMENT_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    shutil.copy2(timeseries_path, work / "player_running_timeseries.csv")
    if running_quality:
        shutil.copy2(Path(running_quality).resolve(), work / "running_quality_report.json")
    (work / "summary.txt").write_text(
        f"共生成 {len(players)} 个已确认球员数据包；排除 {identity_audit['excluded_conflict']} 个冲突ID、"
        f"{identity_audit['excluded_mismatch']} 个不一致ID、{identity_audit['excluded_unreadable']} 个不可读ID。\n",
        encoding="utf-8",
    )
    artifacts = {str(path.relative_to(work)).replace("\\", "/"): sha256(path)
                 for path in sorted(work.rglob("*")) if path.is_file()}
    manifest = {
        "pipeline": "player_card_data_v1", "generated_at": generated_at,
        "identity_audit": identity_audit, "players": sorted(players),
        "event_count": len(normalized_events),
        "inputs": {
            "video": {"path": str(video), "sha256": sha256(video)},
            "mot": ({"path": str(Path(source_mot).resolve()), "sha256": sha256(Path(source_mot).resolve())}
                    if source_mot else None),
            "numbers": {"path": str(Path(numbers).resolve()), "sha256": sha256(Path(numbers).resolve())},
            "events": {"path": str(events_path), "sha256": sha256(events_path)},
            "calibration": {"path": str(calibration_path), "sha256": sha256(calibration_path)},
        },
        "artifacts": artifacts,
    }
    (work / "package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    work.rename(output)
    return manifest
