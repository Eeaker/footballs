from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np


def parse_mot(path: Path) -> dict[int, list[dict]]:
    tracks: dict[int, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.rstrip().split(",")
            if len(parts) < 7:
                raise ValueError(f"{path}:{line_number}: invalid MOT row")
            frame, gid = int(float(parts[0])) - 1, int(float(parts[1]))
            x, y, w, h, confidence = map(float, parts[2:7])
            tracks[gid].append({
                "frame": frame, "x": x, "y": y, "w": w, "h": h,
                "confidence": confidence,
            })
    for rows in tracks.values():
        rows.sort(key=lambda row: row["frame"])
    return dict(tracks)


def choose_uniform(rows: list[dict], count: int) -> list[dict]:
    """Choose one clear observation from each equal temporal quantile."""
    if len(rows) <= count:
        return rows
    chosen = []
    edges = np.linspace(0, len(rows), count + 1).astype(int)
    for index in range(count):
        start, end = edges[index], max(edges[index + 1], edges[index] + 1)
        bucket = rows[start:end]
        center_frame = 0.5 * (bucket[0]["frame"] + bucket[-1]["frame"])
        max_area = max(row["w"] * row["h"] for row in bucket) or 1.0
        span = max(bucket[-1]["frame"] - bucket[0]["frame"], 1)
        def quality(row: dict) -> float:
            temporal = abs(row["frame"] - center_frame) / span
            area = row["w"] * row["h"] / max_area
            return 0.55 * area + 0.35 * row["confidence"] - 0.10 * temporal
        chosen.append(max(bucket, key=quality))
    return chosen


def time_text(seconds: float) -> str:
    minutes = int(seconds // 60)
    remain = seconds - minutes * 60
    return f"{minutes:02d}:{remain:04.1f}"


def visibility_intervals(rows: list[dict], fps: float, merge_gap_seconds: float = 3.0) -> list[list[float]]:
    frames = sorted({int(row["frame"]) for row in rows})
    gap = max(1, int(round(merge_gap_seconds * fps)))
    intervals = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame - previous > gap:
            intervals.append([round(start / fps, 3), round(previous / fps, 3)])
            start = frame
        previous = frame
    intervals.append([round(start / fps, 3), round(previous / fps, 3)])
    return intervals


def crop_person(frame: np.ndarray, row: dict) -> np.ndarray:
    height, width = frame.shape[:2]
    x, y, w, h = row["x"], row["y"], row["w"], row["h"]
    pad_x, pad_top, pad_bottom = 0.35 * w, 0.14 * h, 0.12 * h
    x1 = max(0, int(round(x - pad_x)))
    y1 = max(0, int(round(y - pad_top)))
    x2 = min(width, int(round(x + w + pad_x)))
    y2 = min(height, int(round(y + h + pad_bottom)))
    return frame[y1:y2, x1:x2].copy()


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 25, np.uint8)
    if image.size == 0:
        return canvas
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def create_card(gid: int, selected: list[dict], crops: dict[tuple[int, int], np.ndarray],
                rows: list[dict], fps: float, intervals: list[list[float]]) -> np.ndarray:
    columns, rows_count = 4, 3
    cell_w, image_h, label_h, gap = 300, 300, 42, 10
    header_h = 132
    card_w = columns * cell_w + (columns + 1) * gap
    card_h = header_h + rows_count * (image_h + label_h) + (rows_count + 1) * gap
    card = np.full((card_h, card_w, 3), 238, np.uint8)
    cv2.rectangle(card, (0, 0), (card_w - 1, header_h - 1), (31, 39, 53), -1)
    first, last = rows[0]["frame"] / fps, rows[-1]["frame"] / fps
    visible_seconds = len({row["frame"] for row in rows}) / fps
    cv2.putText(card, f"GLOBAL ID {gid:02d}", (24, 46), cv2.FONT_HERSHEY_DUPLEX,
                1.15, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        card,
        f"appearance range {time_text(first)} - {time_text(last)}  |  visible {visible_seconds:.1f}s",
        (24, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (205, 218, 238), 2, cv2.LINE_AA,
    )
    cv2.putText(
        card, f"visibility segments (gap >3s): {len(intervals)}  |  12 temporal bins",
        (24, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (176, 195, 220), 1, cv2.LINE_AA,
    )
    for index, row in enumerate(selected):
        grid_y, grid_x = divmod(index, columns)
        x = gap + grid_x * (cell_w + gap)
        y = header_h + gap + grid_y * (image_h + label_h + gap)
        tile = fit_image(crops[(gid, row["frame"])], cell_w, image_h)
        card[y:y + image_h, x:x + cell_w] = tile
        cv2.rectangle(card, (x, y), (x + cell_w - 1, y + image_h - 1), (85, 94, 108), 2)
        cv2.rectangle(card, (x, y + image_h), (x + cell_w - 1, y + image_h + label_h - 1),
                      (48, 57, 70), -1)
        label = f"{index + 1:02d}  t={time_text(row['frame'] / fps)}  conf={row['confidence']:.2f}"
        cv2.putText(card, label, (x + 9, y + image_h + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one 3x4 identity mosaic for every global ID")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12, choices=range(9, 13))
    parser.add_argument("--wall-columns", type=int, default=4)
    args = parser.parse_args()

    tracks = parse_mot(args.mot)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    selected = {gid: choose_uniform(rows, args.samples) for gid, rows in tracks.items()}
    requests: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for gid, rows in selected.items():
        for row in rows:
            requests[int(row["frame"])].append((gid, row))

    crops: dict[tuple[int, int], np.ndarray] = {}
    for frame_index in sorted(requests):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {frame_index}")
        for gid, row in requests[frame_index]:
            crops[(gid, frame_index)] = crop_person(frame, row)
    capture.release()

    args.outdir.mkdir(parents=True, exist_ok=True)
    cards = []
    manifest = []
    for gid in sorted(tracks):
        intervals = visibility_intervals(tracks[gid], fps)
        card = create_card(gid, selected[gid], crops, tracks[gid], fps, intervals)
        filename = f"global_id_{gid:02d}_identity_mosaic.jpg"
        cv2.imwrite(str(args.outdir / filename), card, [cv2.IMWRITE_JPEG_QUALITY, 94])
        cards.append((gid, card))
        manifest.append({
            "global_id": gid,
            "mosaic_file": filename,
            "sample_count": len(selected[gid]),
            "sample_frames": [row["frame"] for row in selected[gid]],
            "sample_times_seconds": [round(row["frame"] / fps, 3) for row in selected[gid]],
            "first_seen_seconds": round(tracks[gid][0]["frame"] / fps, 3),
            "last_seen_seconds": round(tracks[gid][-1]["frame"] / fps, 3),
            "visible_seconds": round(len({row["frame"] for row in tracks[gid]}) / fps, 3),
            "visibility_intervals_seconds": intervals,
        })

    thumb_w, thumb_h = 420, 390
    wall_columns = max(1, args.wall_columns)
    wall_rows = (len(cards) + wall_columns - 1) // wall_columns
    wall = np.full((wall_rows * thumb_h, wall_columns * thumb_w, 3), 230, np.uint8)
    for index, (gid, card) in enumerate(cards):
        row, column = divmod(index, wall_columns)
        thumb = fit_image(card, thumb_w - 8, thumb_h - 8)
        y, x = row * thumb_h + 4, column * thumb_w + 4
        wall[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
    cv2.imwrite(str(args.outdir / "all_global_ids_identity_wall.jpg"), wall,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    (args.outdir / "identity_mosaics_manifest.json").write_text(
        json.dumps({"video": str(args.video), "fps": fps, "global_id_count": len(cards),
                    "items": manifest}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"global_id_count": len(cards), "mosaics": len(cards),
                      "wall": str(args.outdir / 'all_global_ids_identity_wall.jpg')},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
