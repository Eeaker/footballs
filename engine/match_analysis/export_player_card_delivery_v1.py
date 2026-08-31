from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import yaml
import time as _time


def safe_rmtree(path: Path, attempts: int = 8, delay: float = 0.5) -> None:
    """Windows frequently keeps short-lived exclusive locks on freshly written
    .mp4 files (Explorer preview pane, antivirus, a lingering subprocess).  Retry
    the delete so a transient lock never aborts an otherwise good export run."""
    path = Path(path)
    if not path.exists():
        return
    last: Exception | None = None
    for _ in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as exc:
            last = exc
            _time.sleep(delay)
    # Final attempt: drop whatever is deletable; any still-locked files are
    # overwritten by the regeneration step below.
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        raise last or RuntimeError(f"无法删除目录: {path}")

from analysis_lib.player_card import (
    calculate_player_running, group_timeseries, load_mot_boxes,
    render_heatmap, render_marked_event_clip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出严格兼容《球员卡数据包对接文档_v1.0》的精简交付包")
    parser.add_argument("--internal-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--numbers", type=Path,
        help="可选：覆盖内部清单中的号码/队伍结果，用于把team_N映射为实际队色并排除非球员簇",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="保留已完整球员目录，只重建缺失或含损坏视频的球员并继续导出",
    )
    parser.add_argument(
        "--running-timeseries", type=Path,
        help="可选：使用主结果目录中的球员时序，允许删除内部交付包里的重复副本",
    )
    return parser.parse_args()


def _video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法读取视频: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"视频FPS无效: {path}")
    return frames / fps


def _copy_or_trim_clip(source: Path, destination: Path, maximum_sec: float = 8.0) -> tuple[float, float]:
    duration = _video_duration(source)
    if duration <= maximum_sec + 0.01:
        shutil.copy2(source, destination)
        return 0.0, min(duration, maximum_sec)
    offset = (duration - maximum_sec) / 2.0
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{offset:.6f}",
        "-i", str(source), "-t", f"{maximum_sec:.6f}",
        "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-c:a", "aac", "-movflags", "+faststart", str(destination),
    ]
    subprocess.run(command, check=True)
    return offset, maximum_sec


def _clock(seconds: float) -> str:
    value = max(0, int(round(seconds)))
    return f"{value // 60:02d}:{value % 60:02d}"


def load_unknown_players(numbers_path: Path) -> list[dict]:
    """Return every unresolved player track; only explicit non-players stay excluded."""
    data = json.loads(Path(numbers_path).read_text(encoding="utf-8"))
    players: dict[int, dict] = {}
    buckets = (
        ("excluded_unreadable", "unreadable"),
        ("excluded_conflict", "conflict"),
        ("excluded_mismatch", "mismatch"),
    )
    for bucket, status in buckets:
        for item in data.get(bucket, []):
            team = str(item.get("team") or "").strip().lower()
            # Team clustering deliberately emits run-local labels (team_0,
            # team_1, team_2) until a human maps them to semantic colours.
            # A colour allow-list therefore drops every valid player in a
            # three-cluster run.  Exclude only targets explicitly classified
            # as non-players; an unknown team is still an unresolved player.
            if team in {"exclude", "excluded", "nonplayer", "non_player"}:
                continue
            if not team:
                team = "unknown"
            gid = int(item["global_id"])
            players[gid] = {
                "player_id": f"unknown_{gid}", "global_id": gid,
                "team": team, "status": status,
            }
    return sorted(players.values(), key=lambda item: item["global_id"])


def _delivery_running(player_id: str, running_source: dict) -> dict:
    summary_source = running_source["summary"]
    heatmap_source = running_source["heatmap"]
    return {
        "player_id": player_id,
        "summary": {
            "total_distance_m": summary_source["total_distance_m"],
            "sprint_count": summary_source["sprint_count"],
            "max_speed_ms": summary_source["speed_p95_mps"],
            "high_speed_distance_m": summary_source["high_speed_distance_m"],
            "playing_time_sec": summary_source["tracked_visible_time_sec"],
        },
        "heatmap": {
            "file_path": "heatmap.png",
            "pitch_size_m": heatmap_source["pitch_size_m"],
            "resolution": heatmap_source["resolution"],
        },
        "raw_data": {
            "mot_file": running_source["raw_data"]["mot_file"],
            "frame_count": summary_source["valid_frame_count"],
        },
    }


def player_delivery_complete(player_dir: Path) -> bool:
    """A resumable checkpoint is valid only when every declared clip exists."""
    required = ("identity.yaml", "running.json", "heatmap.png", "events_for_annotation.json")
    if any(not (player_dir / name).is_file() or (player_dir / name).stat().st_size == 0 for name in required):
        return False
    try:
        annotation = json.loads((player_dir / "events_for_annotation.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    events = annotation.get("events") or []
    if int(annotation.get("total_events", -1)) != len(events):
        return False
    for event in events:
        relative = str(event.get("video_file") or "")
        video = player_dir / relative
        metadata = video.with_suffix(".json")
        # OpenCV can leave a 44-byte MP4 header when the encoder or disk fails.
        if not video.is_file() or video.stat().st_size <= 1024:
            return False
        if not metadata.is_file() or metadata.stat().st_size == 0:
            return False
    return True


def export_delivery(
    internal_package: Path, output: Path, numbers_override: Path | None = None,
    *, resume: bool = False, running_timeseries_override: Path | None = None,
) -> None:
    internal_package = internal_package.resolve()
    output = output.resolve()
    if output.exists() and not resume:
        raise FileExistsError(f"output must not exist: {output}")
    if not internal_package.is_dir():
        raise FileNotFoundError(internal_package)

    root_identity = yaml.safe_load((internal_package / "identity.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((internal_package / "package_manifest.json").read_text(encoding="utf-8"))
    numbers_path = (
        numbers_override.resolve() if numbers_override is not None
        else Path(manifest["inputs"]["numbers"]["path"])
    )
    unknown_players = load_unknown_players(numbers_path)
    player_ids = [item["player_id"] for item in root_identity["players"]]
    total_players = len(player_ids) + len(unknown_players)
    completed_players = 0
    output.mkdir(parents=True, exist_ok=resume)
    print(f"FT player cards: players=0/{total_players}", flush=True)

    for player_id in sorted(player_ids):
        source_dir = internal_package / player_id
        destination_dir = output / player_id
        if resume and player_delivery_complete(destination_dir):
            completed_players += 1
            print(f"FT player cards: players={completed_players}/{total_players}", flush=True)
            continue
        if destination_dir.exists():
            safe_rmtree(destination_dir)
        highlights_dir = destination_dir / "highlights"
        highlights_dir.mkdir(parents=True)

        identity_source = yaml.safe_load((source_dir / "identity.yaml").read_text(encoding="utf-8"))
        player = identity_source["player"]
        resolution = player.get("identity_resolution", {})
        metric_gids = resolution.get("metric_global_ids") or player.get("global_ids") or []
        if not metric_gids:
            raise ValueError(f"球员没有可交付global_id: {player_id}")
        identity = {
            "player": {
                "global_id": int(metric_gids[0]),
                "team": player["team"],
                "jersey_number": int(player["jersey_number"]),
                "status": player["status"],
            },
            "source": identity_source["source"],
            "metadata": identity_source["metadata"],
        }
        (destination_dir / "identity.yaml").write_text(
            yaml.safe_dump(identity, allow_unicode=True, sort_keys=False), encoding="utf-8",
        )

        running_source = json.loads((source_dir / "running.json").read_text(encoding="utf-8"))
        # v1.0 explicitly defines max_speed_ms as the valid-speed P95 despite
        # the historical field-name typo. Preserve that contract here.
        running = _delivery_running(player_id, running_source)
        (destination_dir / "running.json").write_text(
            json.dumps(running, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        shutil.copy2(source_dir / "heatmap.png", destination_dir / "heatmap.png")

        source_events = json.loads(
            (source_dir / "events_for_annotation.json").read_text(encoding="utf-8")
        )
        delivery_events = []
        for sequence, event in enumerate(source_events.get("events", []), 1):
            delivery_event_id = f"ev{sequence:03d}"
            filename = f"{player_id}_{delivery_event_id}.mp4"
            source_clip = source_dir / event["video_file"]
            offset, clip_duration = _copy_or_trim_clip(source_clip, highlights_dir / filename)
            source_start = float(event["start_time"]) + offset
            source_end = source_start + clip_duration

            source_metadata = source_dir / "highlights" / f"{source_clip.stem}.json"
            metadata_source = json.loads(source_metadata.read_text(encoding="utf-8"))
            highlight_metadata = {
                "event_id": delivery_event_id,
                "file_name": filename,
                "player_id": player_id,
                "time_range": {
                    "start_sec": round(source_start, 3),
                    "end_sec": round(source_end, 3),
                    "duration_sec": round(clip_duration, 3),
                },
                "event_type": event["event_type"],
                "source_video": metadata_source["source_video"],
                "confidence": event.get("confidence"),
            }
            (highlights_dir / f"{Path(filename).stem}.json").write_text(
                json.dumps(highlight_metadata, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            actor_status = str(event.get("actor_assignment_status") or "")
            confidence = float(event.get("confidence") or 0.0)
            if actor_status.startswith("provisional") or confidence < 0.5:
                description = (
                    f"算法候选片段（{event['event_type']}），球员归属与具体行为请观看视频后人工确认"
                )
            else:
                description = event.get("description") or ""
            delivery_events.append({
                "event_id": delivery_event_id,
                "sequence": sequence,
                "video_file": f"highlights/{filename}",
                "source_time": f"{_clock(source_start)}-{_clock(source_end)}",
                "source_time_sec": [round(source_start, 3), round(source_end, 3)],
                "event_type_hint": event["event_type"],
                "description": description,
                "status": "pending_semantic_label",
                "semantic_label": None,
            })

        annotation = {
            "player_id": player_id,
            "player_identity": {
                "team": player["team"], "jersey_number": int(player["jersey_number"]),
            },
            "total_events": len(delivery_events),
            "events": delivery_events,
            "annotation_guide": {
                "dimensions": ["体", "技", "观", "决", "战", "心", "智", "位"],
                "instruction": "观看每个视频片段，在Excel台账中填写八维评分（1-10分）和文字观察",
                "output_file": f"{player_id}_semantic_labels.xlsx",
            },
        }
        (destination_dir / "events_for_annotation.json").write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        completed_players += 1
        print(f"FT player cards: players={completed_players}/{total_players}", flush=True)

    timeseries_path = (
        running_timeseries_override.resolve()
        if running_timeseries_override is not None
        else internal_package / "player_running_timeseries.csv"
    )
    if not timeseries_path.is_file():
        raise FileNotFoundError(f"球员位置时序不存在: {timeseries_path}")
    timeseries = group_timeseries(timeseries_path)
    calibration_path = Path(manifest["inputs"]["calibration"]["path"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    bounds = calibration["field_bounds_m"]
    video = Path(manifest["inputs"]["video"]["path"])
    mot_input = manifest.get("inputs", {}).get("mot")
    if unknown_players and not mot_input:
        raise ValueError(
            "正式交付包包含未识别球员，需要在生成球员卡时传入 --mot；"
            "当前 package_manifest.json 的 inputs.mot 为空"
        )
    mot = Path(mot_input["path"]) if mot_input else None
    if mot is not None and not mot.is_file():
        raise FileNotFoundError(f"正式交付包的 MOT 不存在: {mot}")
    mot_boxes = load_mot_boxes(mot) if mot is not None else {}
    video_duration = _video_duration(video)
    video_meta = calibration["video_metadata"]
    fps = float(video_meta["proc_fps"])
    root_events_data = json.loads((internal_package / "events_for_annotation.json").read_text(encoding="utf-8"))
    root_events = root_events_data.get("events", [])
    generated_at = manifest["generated_at"]

    for unknown in unknown_players:
        player_id, gid, team = unknown["player_id"], unknown["global_id"], unknown["team"]
        identity_status = unknown["status"]
        destination_dir = output / player_id
        if resume and player_delivery_complete(destination_dir):
            completed_players += 1
            print(f"FT player cards: players={completed_players}/{total_players}", flush=True)
            continue
        if destination_dir.exists():
            safe_rmtree(destination_dir)
        highlights_dir = destination_dir / "highlights"
        highlights_dir.mkdir(parents=True)
        identity = {
            "player": {
                "global_id": gid, "team": team,
                "jersey_number": None, "status": identity_status,
            },
            "source": {
                "video_file": video.name,
                "video_duration_sec": round(video_duration, 3),
                "processed_frames": int(video_meta["proc_total_frames"]),
            },
            "metadata": {"generated_at": generated_at, "generated_by": "generate_player_card.py"},
        }
        (destination_dir / "identity.yaml").write_text(
            yaml.safe_dump(identity, allow_unicode=True, sort_keys=False), encoding="utf-8",
        )

        rows = timeseries.get(gid, [])
        running_summary, heatmap_rows = calculate_player_running(rows, fps)
        heatmap = render_heatmap(heatmap_rows, bounds, destination_dir / "heatmap.png")
        running_source = {
            "summary": running_summary,
            "heatmap": heatmap,
            "raw_data": {"mot_file": mot.name if mot is not None else None},
        }
        (destination_dir / "running.json").write_text(
            json.dumps(_delivery_running(player_id, running_source), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        player_events = []
        candidates = [event for event in root_events
                      if event.get("primary_global_id") not in (None, "")
                      and int(event["primary_global_id"]) == gid]
        for sequence, event in enumerate(candidates, 1):
            event_id = f"ev{sequence:03d}"
            filename = f"{player_id}_{event_id}.mp4"
            centre = float(event.get("anchor_time") or
                           ((float(event["start_time"]) + float(event["end_time"])) / 2.0))
            # Leave headroom for codec frame rounding so ffprobe remains <=8 s.
            target_duration = 7.75
            start_sec = max(0.0, centre - target_duration / 2.0)
            end_sec = min(video_duration, start_sec + target_duration)
            if end_sec - start_sec < 4.0:
                start_sec = max(0.0, end_sec - target_duration)
            render_marked_event_clip(
                video=video, destination=highlights_dir / filename,
                start_sec=start_sec, end_sec=end_sec, target_gid=gid,
                target_boxes=mot_boxes.get(gid, {}), team=team, jersey_number=None,
            )
            duration = end_sec - start_sec
            metadata = {
                "event_id": event_id, "file_name": filename, "player_id": player_id,
                "time_range": {
                    "start_sec": round(start_sec, 3), "end_sec": round(end_sec, 3),
                    "duration_sec": round(duration, 3),
                },
                "event_type": event["event_type"], "source_video": video.name,
                "confidence": event.get("confidence"),
            }
            (highlights_dir / f"{Path(filename).stem}.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            player_events.append({
                "event_id": event_id, "sequence": sequence,
                "video_file": f"highlights/{filename}",
                "source_time": f"{_clock(start_sec)}-{_clock(end_sec)}",
                "source_time_sec": [round(start_sec, 3), round(end_sec, 3)],
                "event_type_hint": event["event_type"],
                "description": (
                    f"算法候选片段（{event['event_type']}），号码未确认（{identity_status}），"
                    "请观看视频后人工确认球员身份与具体行为"
                ),
                "status": "pending_semantic_label", "semantic_label": None,
            })
        annotation = {
            "player_id": player_id,
            "player_identity": {"team": team, "jersey_number": None},
            "total_events": len(player_events), "events": player_events,
            "annotation_guide": {
                "dimensions": ["体", "技", "观", "决", "战", "心", "智", "位"],
                "instruction": "观看每个视频片段，在Excel台账中填写八维评分（1-10分）和文字观察",
                "output_file": f"{player_id}_semantic_labels.xlsx",
            },
        }
        (destination_dir / "events_for_annotation.json").write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        completed_players += 1
        print(f"FT player cards: players={completed_players}/{total_players}", flush=True)

    audit = root_identity.get("audit", {})
    unresolved_counts: dict[str, int] = {}
    for item in unknown_players:
        unresolved_counts[item["status"]] = unresolved_counts.get(item["status"], 0) + 1
    nonplayer_unreadable = max(
        0, int(audit.get("excluded_unreadable", 0)) - unresolved_counts.get("unreadable", 0)
    )
    (output / "summary.txt").write_text(
        f"共生成 {len(player_ids)} 个已确认球员数据包和 {len(unknown_players)} 个未识别球员数据包；"
        f"其中 {unresolved_counts.get('conflict', 0)} 个号码冲突、"
        f"{unresolved_counts.get('mismatch', 0)} 个号码不一致，均已按 unknown 保留；"
        f"{nonplayer_unreadable} 个非球员目标已排除。\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    export_delivery(
        args.internal_package, args.output, args.numbers, resume=args.resume,
        running_timeseries_override=args.running_timeseries,
    )
    print(json.dumps({"output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
