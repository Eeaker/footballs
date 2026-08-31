from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import cv2
import numpy as np

from analysis_lib.io import MOTBox
from analysis_lib.tracking_adapter import representative_hsv, suggested_color_name, torso_feature
from analysis_lib.teams import assign_teams_kmeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare auditable player-card inputs for the 2026-07-24 video")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--event-index", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_mot_deduplicated(path: Path) -> tuple[list[MOTBox], int]:
    """Keep the highest-confidence box for duplicate (frame, global_id) rows."""
    selected: dict[tuple[int, int], MOTBox] = {}
    duplicate_rows = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        values = line.split(",")
        row = MOTBox(
            int(float(values[0])) - 1, int(float(values[1])),
            *map(float, values[2:7]),
        )
        key = (row.frame_proc, row.global_id)
        previous = selected.get(key)
        if previous is not None:
            duplicate_rows += 1
        if previous is None or row.confidence > previous.confidence:
            selected[key] = row
    return sorted(selected.values(), key=lambda item: (item.frame_proc, item.global_id)), duplicate_rows


def _sample_cluster_colours(video: Path, rows: list[MOTBox], team_map: dict[int, str]) -> dict[str, dict]:
    by_id: dict[int, list[MOTBox]] = defaultdict(list)
    for row in rows:
        by_id[row.global_id].append(row)
    requests: dict[int, list[MOTBox]] = defaultdict(list)
    for gid, track in by_id.items():
        track.sort(key=lambda item: item.frame_proc)
        for index in np.linspace(0, len(track) - 1, min(8, len(track)), dtype=int):
            requests[track[int(index)].frame_proc].append(track[int(index)])
    crops: dict[str, list[np.ndarray]] = defaultdict(list)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    for frame_index in sorted(requests):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        for box in requests[frame_index]:
            item = torso_feature(frame, (box.x, box.y, box.x + box.width, box.y + box.height))
            if item is not None and team_map.get(box.global_id) != "unassigned":
                crops[team_map[box.global_id]].append(item[1])
    cap.release()
    result = {}
    for cluster, items in sorted(crops.items()):
        hsv = representative_hsv(items)
        result[cluster] = {
            "representative_hsv": [round(float(value), 3) for value in hsv],
            "suggested_colour": suggested_color_name(hsv),
            "crop_count": len(items),
        }
    return result


def _semantic_team_map(colours: dict[str, dict]) -> dict[str, str]:
    result: dict[str, str] = {}
    used: set[str] = set()
    for cluster, detail in colours.items():
        colour = str(detail["suggested_colour"])
        if colour in {"yellow", "blue"} and colour not in used:
            result[cluster] = colour
            used.add(colour)
    remaining_clusters = [cluster for cluster in sorted(colours) if cluster not in result]
    remaining_colours = [colour for colour in ("yellow", "blue") if colour not in used]
    for cluster, colour in zip(remaining_clusters, remaining_colours):
        result[cluster] = colour
    if set(result.values()) != {"yellow", "blue"}:
        raise ValueError(f"could not resolve yellow/blue team semantics: {colours}")
    return result


def _motion_proxy_actor(rows: list[MOTBox], anchor_frame: int, radius: int = 6) -> tuple[int | None, float, dict]:
    by_id: dict[int, list[MOTBox]] = defaultdict(list)
    for row in rows:
        if anchor_frame - radius - 3 <= row.frame_proc <= anchor_frame + radius:
            by_id[row.global_id].append(row)
    scores: list[tuple[float, int, int]] = []
    for gid, track in by_id.items():
        track.sort(key=lambda item: item.frame_proc)
        local: list[float] = []
        for left, right in zip(track, track[1:]):
            delta = right.frame_proc - left.frame_proc
            if delta <= 0 or delta > 3:
                continue
            x0, y0 = left.x + left.width / 2.0, left.y + left.height / 2.0
            x1, y1 = right.x + right.width / 2.0, right.y + right.height / 2.0
            local.append(float(np.hypot(x1 - x0, y1 - y0) / delta * 30.0))
        if local:
            scores.append((float(np.median(local)), gid, len(local)))
    scores.sort(reverse=True)
    if not scores:
        return None, 0.0, {"candidates": 0, "margin_px_per_sec": 0.0}
    top = scores[0]
    second = scores[1][0] if len(scores) > 1 else 0.0
    margin = max(0.0, top[0] - second)
    confidence = min(0.49, 0.20 + 0.29 * margin / max(top[0], 1e-6))
    return top[1], round(confidence, 3), {
        "candidates": len(scores), "top_speed_px_per_sec": round(top[0], 3),
        "margin_px_per_sec": round(margin, 3), "local_steps": top[2],
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output must not exist: {output}")
    output.mkdir(parents=True)
    rows, duplicate_rows = _read_mot_deduplicated(args.mot)
    team_map, team_diagnostics = assign_teams_kmeans(args.video, rows, n_clusters=2)
    colours = _sample_cluster_colours(args.video, rows, team_map)
    semantics = _semantic_team_map(colours)
    resolved_teams = {gid: semantics.get(cluster, "unassigned") for gid, cluster in team_map.items()}

    counts = defaultdict(int)
    for row in rows:
        counts[row.global_id] += 1
    unreadable = []
    excluded_nonplayer = []
    for gid in sorted(counts):
        team = resolved_teams.get(gid, "unassigned")
        item = {
            "global_id": gid, "team": team, "reason": "number_ocr_unavailable_for_this_video",
            "mot_detection_count": counts[gid], "status": "unreadable",
        }
        if team in {"yellow", "blue"}:
            unreadable.append(item)
        else:
            item["reason"] = "no_valid_team_jersey_sample"
            excluded_nonplayer.append(item)
    numbers = {
        "schema_version": "clip-eligibility-v1",
        "data_status": "number_ocr_unavailable_no_cross_video_reuse",
        "eligible_confirmed": [], "excluded_conflict": [], "excluded_mismatch": [],
        "excluded_unreadable": unreadable, "excluded_nonplayer": excluded_nonplayer,
        "team_scope": ["yellow", "blue"],
        "note": "旧U12号码结果与本视频哈希不一致；本文件不伪造号码，所有同场追踪ID按未识别交付。",
    }
    (output / "clip_eligibility.json").write_text(
        json.dumps(numbers, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    index = json.loads(args.event_index.read_text(encoding="utf-8"))
    by_event = {int(item["event_id"]): item for item in index}
    clips = []
    for clip in sorted(args.clips.glob("event_*.mp4")):
        match = re.match(r"event_(\d+)_", clip.name)
        if not match:
            continue
        event_id = int(match.group(1))
        if event_id not in by_event:
            continue
        source = by_event[event_id]
        gid, confidence, proxy = _motion_proxy_actor(rows, int(source["event_frame_proc"]))
        clips.append({
            "event_id": f"source_event_{event_id:03d}",
            "start_time": round(int(source["start_frame_proc"]) / 30.0, 3),
            "end_time": round((int(source["end_frame_proc"]) + 1) / 30.0, 3),
            "primary_global_id": gid, "secondary_global_id": None,
            "event_type": source["event_type"], "video_anchor_path": str(clip.resolve()),
            "jersey_number": None, "confidence": confidence,
            "actor_assignment_status": "provisional_player_motion_proxy",
            "actor_assignment_evidence": proxy,
            "description": "事件类型和球员归属均为算法候选，等待人工复核。",
        })
    events = {
        "schema_version": "events-for-annotation-v1", "total_events": len(clips), "events": clips,
        "provenance": {
            "source_event_index": str(args.event_index.resolve()),
            "actor_policy": "adjacent-frame centre-speed proxy; not possession truth",
        },
    }
    (output / "events_for_annotation.json").write_text(
        json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    audit = {
        "mot_rows_after_deduplication": len(rows), "duplicate_mot_rows_removed": duplicate_rows,
        "global_ids": len(counts), "unreadable_players": len(unreadable),
        "excluded_unassigned": len(excluded_nonplayer), "event_clips": len(clips),
        "cluster_colours": colours, "cluster_to_team": semantics,
        "team_diagnostics": team_diagnostics,
    }
    (output / "input_preparation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **{key: audit[key] for key in (
        "global_ids", "unreadable_players", "excluded_unassigned", "event_clips")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
