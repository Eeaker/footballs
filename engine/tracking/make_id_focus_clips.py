from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2


def read_mot(path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    frames = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.rstrip().split(",")
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_number} 非法MOT行")
            proc_frame = int(float(parts[0])) - 1
            frames[proc_frame].append((
                int(float(parts[1])), float(parts[2]), float(parts[3]),
                float(parts[4]), float(parts[5]),
            ))
    return dict(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成带目标global_id高亮的事件切片")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True, help="JSON事件索引（允许UTF-8 BOM）")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="输出JSON清单；扩展名必须为.json")
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--include-review", action="store_true")
    args = parser.parse_args()
    if args.vid_stride < 1:
        parser.error("vid-stride必须>=1")
    if args.manifest is not None and args.manifest.suffix.lower() != ".json":
        parser.error("manifest是JSON文件，扩展名必须为.json")
    events = json.loads(args.events.read_text(encoding="utf-8-sig"))
    events = [
        row for row in events
        if row.get("primary_global_id") is not None
        and (args.include_review or row.get("actor_attribution_status") == "auto")
    ]
    args.outdir.mkdir(parents=True, exist_ok=True)
    mot = read_mot(args.mot)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")
    raw_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    proc_fps = raw_fps / args.vid_stride
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    active = {}
    output_rows = []
    for row in events:
        event_id = int(row["event_id"])
        gid = int(row["primary_global_id"])
        name = f"event_{event_id:04d}_gid_{gid:03d}.mp4"
        writer = cv2.VideoWriter(
            str(args.outdir / name), cv2.VideoWriter_fourcc(*"mp4v"), proc_fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"无法创建切片: {name}")
        active[event_id] = (writer, row, gid, name)

    raw_index = -1
    proc_index = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_index += 1
        if (raw_index + 1) % args.vid_stride != 0:
            continue
        proc_index += 1
        boxes = mot.get(proc_index, [])
        for writer, row, gid, _ in active.values():
            if not (int(row["start_frame_proc"]) <= proc_index < int(row["end_frame_proc"])):
                continue
            canvas = frame.copy()
            target_seen = False
            for box_gid, x, y, w, h in boxes:
                color = (40, 220, 40) if box_gid == gid else (110, 110, 110)
                thickness = 4 if box_gid == gid else 1
                p1 = (max(0, int(round(x))), max(0, int(round(y))))
                p2 = (min(width - 1, int(round(x + w))), min(height - 1, int(round(y + h))))
                cv2.rectangle(canvas, p1, p2, color, thickness)
                if box_gid == gid:
                    target_seen = True
                    cv2.putText(canvas, f"TARGET ID {gid}", (p1[0], max(25, p1[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
            status = f"event {row['event_id']}  target global_id={gid}"
            if not target_seen:
                status += "  [target not visible]"
            cv2.putText(canvas, status, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(canvas)
    cap.release()
    for writer, row, gid, name in active.values():
        writer.release()
        output_rows.append({
            "event_id": row["event_id"], "clip_id": f"event_{int(row['event_id']):04d}",
            "clip_file": name, "global_id": gid,
            "primary_global_id": gid,
            "anchor_offset_seconds": round(
                (int(row["event_frame_proc"]) - int(row["start_frame_proc"])) / proc_fps, 3
            ),
            "duration_seconds": round(
                (int(row["end_frame_proc"]) - int(row["start_frame_proc"])) / proc_fps, 3
            ),
            "actor_attribution_status": row.get("actor_attribution_status"),
            "base_event_type": row.get("base_event_type"),
        })
    manifest = args.manifest or args.outdir / "id_focus_clips.json"
    manifest.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"clips": len(output_rows), "manifest": str(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
