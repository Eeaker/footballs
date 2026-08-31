from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2

DEFAULT_MAX_EVENTS = 5
DEFAULT_MAX_DURATION_SECONDS = 150.0
DEFAULT_PER_EVENT_DURATION_SECONDS = 30.0
SOURCE_WINDOW_BEFORE_SECONDS = 15.0
SOURCE_WINDOW_AFTER_SECONDS = 15.0


def load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["events"] if isinstance(data, dict) and "events" in data else data


def matches(row: dict, args: argparse.Namespace) -> bool:
    identity = row.get("player_id") if args.player_id is not None else row.get(
        "primary_global_id", row.get("global_id")
    )
    requested = args.player_id if args.player_id is not None else args.global_id
    if requested is not None and identity != requested:
        return False
    if args.dimension and row.get("main_dimension") != args.dimension:
        return False
    labels = set(row.get("behavior_labels", []))
    if args.label and args.label not in labels:
        return False
    if not args.include_review and (row.get("review_required") or row.get("actor_attribution_status") == "review"):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="按ID/维度/标签生成定长足球集锦")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--global-id", type=int)
    parser.add_argument("--player-id", type=int)
    parser.add_argument("--dimension")
    parser.add_argument("--label")
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION_SECONDS,
                        help="5个事件各30秒，默认拼接总长约150秒")
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS,
                        help="默认最多使用5个已复核事件")
    parser.add_argument("--per-event-duration", type=float, default=DEFAULT_PER_EVENT_DURATION_SECONDS,
                        help="每个事件完整保留事件点前15秒和后15秒，共30秒")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--include-review", action="store_true")
    parser.add_argument(
        "--order", choices=["chronological", "manifest", "confidence", "salience"],
        default="chronological",
        help="默认按比赛时间，不把模型置信度或事件显著度当作球员评分",
    )
    args = parser.parse_args()
    if args.max_duration <= 0 or args.per_event_duration <= 0 or args.fps <= 0:
        parser.error("时长和fps必须大于0")
    if args.max_events <= 0:
        parser.error("max-events必须大于0")

    rows = [row for row in load_rows(args.manifest) if matches(row, args)]
    if args.order == "chronological":
        rows.sort(key=lambda row: (
            float(row.get("anchor_seconds", row.get("event_frame_proc", float("inf")))),
            str(row.get("clip_id", row.get("event_id", ""))),
        ))
    elif args.order == "confidence":
        rows.sort(key=lambda row: -float(row.get("confidence", 0.0)))
    elif args.order == "salience":
        rows.sort(key=lambda row: -float(row.get("score", 0.0)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频: {args.output}")

    max_frames = int(round(args.max_duration * args.fps))
    per_event_frames = int(round(args.per_event_duration * args.fps))
    written = 0
    selected = []
    try:
        for row in rows:
            if written >= max_frames or len(selected) >= args.max_events:
                break
            remaining = max_frames - written
            if remaining < min(per_event_frames, int(round(args.fps))):
                break
            clip_name = row.get("clip_file") or row.get("clip_path")
            if not clip_name:
                continue
            clip_path = Path(clip_name)
            if not clip_path.is_absolute():
                clip_path = args.clips_dir / clip_path
            cap = cv2.VideoCapture(str(clip_path))
            if not cap.isOpened():
                continue
            source_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
            source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            desired = min(per_event_frames, remaining)
            anchor_offset = row.get("anchor_offset_seconds")
            if anchor_offset is None and "anchor_seconds" in row and "start_seconds" in row:
                anchor_offset = float(row["anchor_seconds"]) - float(row["start_seconds"])
            if anchor_offset is None:
                anchor_source = source_frames / 2
            else:
                anchor_source = float(anchor_offset) * source_fps
            source_needed = desired / args.fps * source_fps
            source_start = max(0, int(round(anchor_source - 0.5 * source_needed)))
            source_end = min(source_frames, int(math.ceil(source_start + source_needed)) + 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_start)
            source_buffer = []
            for _ in range(max(0, source_end - source_start)):
                ok, frame = cap.read()
                if not ok:
                    break
                source_buffer.append(frame)
            cap.release()
            event_written = 0
            if source_buffer:
                for out_index in range(desired):
                    source_index = min(
                        len(source_buffer) - 1,
                        int(round(out_index * source_fps / args.fps)),
                    )
                    frame = cv2.resize(
                        source_buffer[source_index], (args.width, args.height),
                        interpolation=cv2.INTER_AREA,
                    )
                    writer.write(frame)
                    written += 1
                    event_written += 1
            if event_written:
                selected.append({
                    "event_id": row.get("event_id"), "clip_id": row.get("clip_id"),
                    "clip_file": str(clip_name), "frames": event_written,
                    "duration_seconds": round(event_written / args.fps, 3),
                })
    finally:
        writer.release()

    output_manifest = args.output_manifest or args.output.with_suffix(".json")
    output_manifest.write_text(json.dumps({
        "output_video": str(args.output), "fps": args.fps,
        "duration_seconds": round(written / args.fps, 3),
        "highlight_policy": {
            "candidate_events": args.max_events,
            "source_window_seconds": {
                "before": SOURCE_WINDOW_BEFORE_SECONDS,
                "after": SOURCE_WINDOW_AFTER_SECONDS,
            },
            "default_compilation_duration_seconds": args.max_duration,
            "seconds_per_event": args.per_event_duration,
        },
        "selected_events": selected,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": len(selected), "duration_seconds": round(written / args.fps, 3)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
