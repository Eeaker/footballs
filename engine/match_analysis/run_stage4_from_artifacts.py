"""Run the repository's original Stage4 logic from preserved MOT + fresh ball CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2

WORKSPACE = Path(__file__).resolve().parent.parent
TRACKING_ROOT = WORKSPACE / "tracking"
if str(TRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKING_ROOT))

from tracking_core import stage4_events


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="从已有 MOT 与足球观测重新运行原 Stage4 事件逻辑")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--ball", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-sec", type=float, default=3.0)
    parser.add_argument("--post-sec", type=float, default=2.0)
    parser.add_argument("--event-percentile", type=float, default=92.0)
    parser.add_argument("--event-min-gap", type=float, default=2.0)
    parser.add_argument("--edge-margin", type=float, default=1.0)
    parser.add_argument("--ball-max-gap", type=int, default=30)
    parser.add_argument("--n-clips", type=int, default=20)
    args = parser.parse_args()

    video, mot, ball, output = (
        args.video.resolve(), args.mot.resolve(), args.ball.resolve(), args.output.resolve()
    )
    if output.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {output}")
    output.mkdir(parents=True)

    detections = []
    gids = set()
    with mot.open("r", encoding="utf-8-sig", newline="") as handle:
        for values in csv.reader(handle):
            if not values:
                continue
            frame, gid = int(float(values[0])) - 1, int(float(values[1]))
            x, y, width, height, confidence = map(float, values[2:7])
            detections.append((frame, gid, x, y, width, height, confidence))
            gids.add(gid)

    ball_positions = {}
    with ball.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("observed", "1")).strip().lower() not in {"1", "true", "yes"}:
                continue
            ball_positions[int(row["frame_proc"])] = (
                float(row["ball_x_px"]), float(row["ball_y_px"]),
                float(row.get("confidence") or 1.0),
            )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    if not detections or not ball_positions or fps <= 0 or total_frames <= 0:
        raise ValueError("MOT、足球观测或视频元数据为空")

    # stage4_events only reads this exact argument subset. Keeping the original
    # defaults here makes the adapter auditable and avoids copying the algorithm.
    args.video = str(video)
    args.outdir = str(output)
    args.vid_stride = 1
    stage4_events(
        args, detections, ball_positions, {gid: gid for gid in gids},
        total_frames, fps, fps,
    )
    manifest = {
        "pipeline": "original_pipeline_stage4_artifact_adapter_v1",
        "video": str(video), "video_sha256": _sha256(video),
        "mot": str(mot), "mot_sha256": _sha256(mot),
        "ball": str(ball), "ball_sha256": _sha256(ball),
        "parameters": {
            "pre_sec": args.pre_sec, "post_sec": args.post_sec,
            "event_percentile": args.event_percentile,
            "event_min_gap": args.event_min_gap, "edge_margin": args.edge_margin,
            "ball_max_gap": args.ball_max_gap, "n_clips": args.n_clips,
        },
    }
    (output / "stage4_adapter_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
