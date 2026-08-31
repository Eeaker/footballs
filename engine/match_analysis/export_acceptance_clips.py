"""Export the deterministic 20-event review clips.

The frame-bound and OpenCV writer loop are migrated from the project-owned
``legacy tracking event exporter``.  This adapter reads match analysis
pass/sample CSV files and overlays the already-exported ball observations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def frame_bounds(
    release_frame: int, receive_frame: int, total_frames: int, fps: float,
    before_release_seconds: float, after_receive_seconds: float,
) -> tuple[int, int]:
    start = max(0, release_frame - int(round(before_release_seconds * fps)))
    end_exclusive = min(total_frames, receive_frame + int(round(after_receive_seconds * fps)))
    return start, end_exclusive


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_events(pass_events: list[dict], sample_rows: list[dict]) -> list[dict]:
    by_id = {int(row["pass_id"]): row for row in pass_events}
    selected = []
    for sample in sample_rows:
        pass_id = int(sample["pass_id"])
        if pass_id not in by_id:
            raise ValueError(f"抽样 pass_id 不在事件表中: {pass_id}")
        selected.append(by_id[pass_id])
    return selected


def read_ball(path: Path) -> dict[int, tuple[float, float]]:
    result = {}
    for row in read_csv(path):
        if str(row.get("observed", "1")).strip().lower() in {"1", "true", "yes"}:
            result[int(row["frame_proc"])] = (float(row["ball_x_px"]), float(row["ball_y_px"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 match analysis 固定20条传球人工验收片段")
    parser.add_argument("--tracking-video", type=Path, required=True)
    parser.add_argument("--ball", type=Path, required=True)
    parser.add_argument("--pass-events", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--before-release-seconds", type=float, default=1.0)
    parser.add_argument("--after-receive-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"输出目录必须不存在，避免覆盖: {args.outdir}")
    if args.before_release_seconds < 0 or args.after_receive_seconds < 0:
        parser.error("前后时长不得小于0")
    events = select_events(read_csv(args.pass_events), read_csv(args.sample))
    ball = read_ball(args.ball)
    capture = cv2.VideoCapture(str(args.tracking_video))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {args.tracking_video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    args.outdir.mkdir(parents=True)
    compilation_path = args.outdir / f"acceptance_{len(events)}_compilation.mp4"
    compilation = cv2.VideoWriter(str(compilation_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not compilation.isOpened():
        raise RuntimeError(f"无法创建集锦: {compilation_path}")
    manifest_rows = []
    try:
        for order, event in enumerate(events, 1):
            pass_id = int(event["pass_id"])
            release = int(event["release_frame_proc"])
            receive = int(event["receive_frame_proc"])
            start, end_exclusive = frame_bounds(
                release, receive, total_frames, fps,
                args.before_release_seconds, args.after_receive_seconds,
            )
            clip_name = f"sample_{order:02d}_pass_{pass_id:03d}.mp4"
            event_label = event.get("classification") or event.get("outcome") or "pass_candidate"
            clip = cv2.VideoWriter(str(args.outdir / clip_name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not clip.isOpened():
                raise RuntimeError(f"无法创建片段: {clip_name}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            written = 0
            try:
                for frame_index in range(start, end_exclusive):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame_index in ball:
                        x, y = ball[frame_index]
                        cv2.circle(frame, (int(round(x)), int(round(y))), 12, (0, 0, 255), 3)
                        cv2.putText(frame, "BALL", (int(x) + 14, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
                    banner = (
                        f"SAMPLE {order:02d}/20  PASS {pass_id:03d}  "
                        f"ID {event['from_global_id']} -> {event['to_global_id']}  {event_label}"
                    )
                    cv2.rectangle(frame, (0, 0), (width, 42), (0, 0, 0), -1)
                    cv2.putText(frame, banner, (15, 29), cv2.FONT_HERSHEY_SIMPLEX, .72, (255, 255, 255), 2)
                    if release <= frame_index <= int(event["receive_confirmed_frame_proc"]):
                        cv2.putText(frame, "HANDOFF WINDOW", (15, 68), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
                    clip.write(frame); compilation.write(frame); written += 1
            finally:
                clip.release()
            manifest_rows.append({
                "sample_order": order, "pass_id": pass_id, "clip_file": clip_name,
                "from_global_id": int(event["from_global_id"]), "to_global_id": int(event["to_global_id"]),
                "model_outcome": event_label, "release_frame_proc": release,
                "receive_frame_proc": receive, "clip_start_seconds": round(start / fps, 3),
                "clip_end_seconds": round(end_exclusive / fps, 3), "duration_seconds": round(written / fps, 3),
                "human_review_required": True,
            })
    finally:
        compilation.release(); capture.release()
    manifest = {
        "status": "pending_human_review",
        "selection_policy": "same deterministic IDs as acceptance_sample_20.csv; up to 20 without padding",
        "tracking_video": str(args.tracking_video.resolve()), "ball_overlay": str(args.ball.resolve()),
        "event_count": len(manifest_rows), "compilation_file": compilation_path.name, "events": manifest_rows,
    }
    (args.outdir / "acceptance_clips_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({"event_count": len(manifest_rows), "output": str(compilation_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
