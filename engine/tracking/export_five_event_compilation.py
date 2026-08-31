from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2


def select_balanced_candidates(events: list[dict], count: int) -> list[dict]:
    """Reproduce Stage-4 stratified discovery without exposing a player score."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        groups[str(event["base_event_type"])].append(event)
    for rows in groups.values():
        rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    types = sorted(groups)
    selected: list[dict] = []
    index = 0
    while len(selected) < count and any(groups[event_type] for event_type in types):
        event_type = types[index % len(types)]
        index += 1
        if groups[event_type]:
            selected.append(groups[event_type].pop(0))
    return sorted(selected, key=lambda row: int(row["event_frame_proc"]))


def frame_bounds(event_frame: int, total_frames: int, fps: float,
                 before_seconds: float, after_seconds: float) -> tuple[int, int]:
    start = max(0, event_frame - int(round(before_seconds * fps)))
    end_exclusive = min(total_frames, event_frame + int(round(after_seconds * fps)))
    return start, end_exclusive


def main() -> None:
    parser = argparse.ArgumentParser(description="从已有事件索引直接导出5×30秒候选集锦")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--before-seconds", type=float, default=15.0)
    parser.add_argument("--after-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if args.count <= 0 or args.before_seconds < 0 or args.after_seconds < 0:
        parser.error("事件数量必须大于0，前后时长不得小于0")

    events = json.loads(args.events.read_text(encoding="utf-8-sig"))
    selected = select_balanced_candidates(events, args.count)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    args.outdir.mkdir(parents=True, exist_ok=True)
    compilation_path = args.outdir / "five_event_candidate_compilation_150s.mp4"
    compilation = cv2.VideoWriter(
        str(compilation_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not compilation.isOpened():
        raise RuntimeError(f"无法创建集锦: {compilation_path}")

    manifest_rows = []
    compilation_frames = 0
    try:
        for order, event in enumerate(selected, 1):
            event_frame = int(event["event_frame_proc"])
            start, end_exclusive = frame_bounds(
                event_frame, total_frames, fps, args.before_seconds, args.after_seconds
            )
            clip_name = f"candidate_{order:02d}_event_{int(event['event_id']):03d}.mp4"
            clip_path = args.outdir / clip_name
            clip = cv2.VideoWriter(
                str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not clip.isOpened():
                raise RuntimeError(f"无法创建候选片段: {clip_path}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            written = 0
            try:
                for _ in range(end_exclusive - start):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    clip.write(frame)
                    compilation.write(frame)
                    written += 1
                    compilation_frames += 1
            finally:
                clip.release()
            manifest_rows.append({
                "compilation_order": order,
                "event_id": int(event["event_id"]),
                "candidate_event_type": event["base_event_type"],
                "candidate_global_id": event.get("primary_global_id"),
                "actor_attribution_status": event.get("actor_attribution_status"),
                "event_time_seconds": round(event_frame / fps, 3),
                "clip_start_seconds": round(start / fps, 3),
                "clip_end_seconds": round(end_exclusive / fps, 3),
                "event_offset_seconds": round((event_frame - start) / fps, 3),
                "duration_seconds": round(written / fps, 3),
                "clip_file": clip_name,
                "human_review_required": True,
            })
    finally:
        compilation.release()
        capture.release()

    manifest = {
        "status": "candidate_not_formally_reviewed",
        "selection_policy": "three_type_stratified_round_robin_then_internal_salience",
        "evaluation_policy": "candidate_labels_plus_human_review_no_multimodal_direct_score",
        "event_count": len(manifest_rows),
        "seconds_before_event": args.before_seconds,
        "seconds_after_event": args.after_seconds,
        "compilation_duration_seconds": round(compilation_frames / fps, 3),
        "compilation_file": compilation_path.name,
        "events": manifest_rows,
    }
    (args.outdir / "five_event_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "event_count": len(manifest_rows),
        "duration_seconds": manifest["compilation_duration_seconds"],
        "output": str(compilation_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
