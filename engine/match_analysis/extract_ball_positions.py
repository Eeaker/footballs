"""Export the ball-observation contract without rerunning player identity.

The detector settings and per-frame highest-confidence football rule are
adapted from the canonical tracking ball-observation stage.
Player tracking/ReID is intentionally omitted so an existing, reviewed MOT can
retain its global_id namespace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_ball_positions(
    *, video: Path, weights: Path, output: Path, device: str = "0",
    imgsz: int = 1280, confidence: float = .25, vid_stride: int = 1,
) -> dict:
    video, weights, output = video.resolve(), weights.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录必须不存在，避免覆盖: {output}")
    if not video.is_file() or not weights.is_file():
        raise FileNotFoundError(f"输入不存在: video={video}, weights={weights}")
    if imgsz < 32 or not 0 < confidence <= 1 or vid_stride < 1:
        raise ValueError("imgsz/confidence/vid_stride 参数无效")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    raw_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    work = output.with_name(output.name + f".work-{int(time.time())}")
    if work.exists():
        raise FileExistsError(work)
    work.mkdir(parents=True)
    started = time.time()
    rows: list[tuple[int, float, float, float]] = []
    frame_proc = -1
    try:
        model = YOLO(str(weights))
        results = model.predict(
            source=str(video), classes=[32], stream=True, conf=confidence,
            iou=.5, vid_stride=vid_stride, imgsz=imgsz, device=device,
            verbose=False,
        )
        for result in results:
            frame_proc += 1
            boxes = result.boxes
            if boxes is not None and len(boxes):
                confidences = boxes.conf.cpu().numpy()
                best = int(confidences.argmax())
                x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy()
                rows.append((frame_proc, (float(x1) + float(x2)) / 2,
                             (float(y1) + float(y2)) / 2, float(confidences[best])))
            if frame_proc % 1000 == 0:
                print(f"ball extraction frame={frame_proc} observed={len(rows)}", flush=True)

        csv_path = work / "ball_positions_observed.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_proc", "ball_x_px", "ball_y_px", "observed", "confidence"])
            for frame, x, y, score in rows:
                writer.writerow([frame, round(x, 3), round(y, 3), 1, round(score, 6)])
        metadata = {
            "pipeline": "tracking_ball_observation_adapter_v1",
            "source_rule": "YOLO class 32; highest-confidence detection per processed frame",
            "video": str(video), "video_sha256": sha256(video),
            "weights": str(weights), "weights_sha256": sha256(weights),
            "raw_fps": raw_fps, "raw_total_frames": raw_frames,
            "frame_width": width, "frame_height": height,
            "vid_stride": vid_stride, "processed_fps": raw_fps / vid_stride,
            "total_processed_frames": frame_proc + 1,
            "observed_ball_frames": len(rows),
            "parameters": {"imgsz": imgsz, "confidence": confidence, "iou": .5, "device": device},
            "elapsed_seconds": round(time.time() - started, 3),
        }
        (work / "ball_extraction_manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        work.rename(output)
        return metadata
    except Exception:
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="从视频导出逐帧足球观测，不改动既有MOT身份")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--vid-stride", type=int, default=1)
    args = parser.parse_args()
    result = extract_ball_positions(
        video=args.video, weights=args.weights, output=args.output, device=args.device,
        imgsz=args.imgsz, confidence=args.confidence, vid_stride=args.vid_stride,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
