from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np


def parse_mot(path: Path) -> dict[int, list[dict]]:
    tracks: dict[int, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.rstrip().split(",")
            if len(parts) < 7:
                raise ValueError(f"{path}:{line_number}: invalid MOT row")
            frame, gid = int(float(parts[0])) - 1, int(float(parts[1]))
            x, y, w, h, confidence = map(float, parts[2:7])
            tracks[gid].append({"frame": frame, "x": x, "y": y, "w": w, "h": h,
                                "confidence": confidence})
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
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def visibility_intervals(rows: list[dict], fps: float, merge_gap_seconds: float = 3.0) -> list[list[float]]:
    frames = sorted({int(row["frame"]) for row in rows})
    gap = max(1, int(round(merge_gap_seconds * fps)))
    intervals, start, previous = [], frames[0], frames[0]
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
    x1 = max(0, int(round(x - .35 * w)))
    y1 = max(0, int(round(y - .14 * h)))
    x2 = min(width, int(round(x + 1.35 * w)))
    y2 = min(height, int(round(y + 1.12 * h)))
    return frame[y1:y2, x1:x2].copy()


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 25, np.uint8)
    if image.size == 0:
        return canvas
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(round(image.shape[1] * scale))),
                                 max(1, int(round(image.shape[0] * scale)))),
                         interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def write_image(path: Path, image: np.ndarray, quality: int) -> None:
    """Write through imencode so Windows paths containing Chinese characters work."""
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"cannot encode image: {path}")
    encoded.tofile(str(path))


def create_card(gid: int, selected: list[dict], crops: dict[tuple[int, int], np.ndarray],
                rows: list[dict], fps: float, intervals: list[list[float]]) -> np.ndarray:
    columns, rows_count, cell_w, image_h, label_h, gap, header_h = 4, 3, 300, 300, 42, 10, 132
    card_w = columns * cell_w + (columns + 1) * gap
    card_h = header_h + rows_count * (image_h + label_h) + (rows_count + 1) * gap
    card = np.full((card_h, card_w, 3), 238, np.uint8)
    cv2.rectangle(card, (0, 0), (card_w - 1, header_h - 1), (31, 39, 53), -1)
    first, last = rows[0]["frame"] / fps, rows[-1]["frame"] / fps
    visible_seconds = len({row["frame"] for row in rows}) / fps
    cv2.putText(card, f"GLOBAL ID {gid:02d}", (24, 46), cv2.FONT_HERSHEY_DUPLEX, 1.15, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(card, f"appearance range {time_text(first)} - {time_text(last)}  |  visible {visible_seconds:.1f}s  |  boxes {len(rows)}", (24, 81), cv2.FONT_HERSHEY_SIMPLEX, .72, (205, 218, 238), 2, cv2.LINE_AA)
    cv2.putText(card, f"visibility segments (gap >3s): {len(intervals)}  |  12 temporal bins", (24, 112), cv2.FONT_HERSHEY_SIMPLEX, .66, (176, 195, 220), 1, cv2.LINE_AA)
    for index, row in enumerate(selected):
        grid_y, grid_x = divmod(index, columns)
        x, y = gap + grid_x * (cell_w + gap), header_h + gap + grid_y * (image_h + label_h + gap)
        card[y:y + image_h, x:x + cell_w] = fit_image(crops[(gid, row["frame"])], cell_w, image_h)
        cv2.rectangle(card, (x, y), (x + cell_w - 1, y + image_h - 1), (85, 94, 108), 2)
        cv2.rectangle(card, (x, y + image_h), (x + cell_w - 1, y + image_h + label_h - 1), (48, 57, 70), -1)
        label = f"{index + 1:02d}  t={time_text(row['frame'] / fps)}  conf={row['confidence']:.2f}"
        cv2.putText(card, label, (x + 9, y + image_h + 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (245, 245, 245), 1, cv2.LINE_AA)
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one 3x4 identity mosaic for every global ID")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12, choices=range(9, 13))
    parser.add_argument("--wall-columns", type=int, default=4)
    parser.add_argument("--wall-page-size", type=int, default=160)
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
    # Decode once in order. Hundreds of random H.264 seeks make large, strict
    # association walls needlessly slow and can land on decoder-dependent
    # neighbouring frames.
    final_requested_frame = max(requests, default=-1)
    for frame_index in range(final_requested_frame + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {frame_index}")
        for gid, row in requests.get(frame_index, []):
            crops[(gid, frame_index)] = crop_person(frame, row)
    capture.release()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cards, manifest = [], []
    for gid in sorted(tracks):
        intervals = visibility_intervals(tracks[gid], fps)
        card = create_card(gid, selected[gid], crops, tracks[gid], fps, intervals)
        filename = f"global_id_{gid:02d}_identity_mosaic.jpg"
        write_image(args.outdir / filename, card, 94)
        # Keep only filenames here. Holding hundreds of full-resolution cards
        # at once can consume several GB when singleton retention is enabled.
        cards.append((gid, filename))
        manifest.append({"global_id": gid, "mosaic_file": filename, "sample_count": len(selected[gid]),
                         "sample_frames": [row["frame"] for row in selected[gid]],
                         "sample_times_seconds": [round(row["frame"] / fps, 3) for row in selected[gid]],
                         "first_seen_seconds": round(tracks[gid][0]["frame"] / fps, 3),
                         "last_seen_seconds": round(tracks[gid][-1]["frame"] / fps, 3),
                         "visible_seconds": round(len({row["frame"] for row in tracks[gid]}) / fps, 3),
                         "visibility_intervals_seconds": intervals})
    thumb_w, thumb_h = 420, 390
    wall_columns = max(1, args.wall_columns)
    wall_header_h = 88
    max_label = max(tracks) if tracks else -1
    page_size = max(wall_columns, int(args.wall_page_size))
    pages = []
    for page_index, start in enumerate(range(0, len(cards), page_size), 1):
        page_cards = cards[start:start + page_size]
        rows = (len(page_cards) + wall_columns - 1) // wall_columns
        wall = np.full((wall_header_h + rows * thumb_h, wall_columns * thumb_w, 3), 230, np.uint8)
        cv2.rectangle(wall, (0, 0), (wall.shape[1] - 1, wall_header_h - 1), (31, 39, 53), -1)
        cv2.putText(wall, f"RETAINED GLOBAL IDs: {len(cards)}  |  PAGE {page_index}  |  MAX LABEL: {max_label}",
                    (22, 38), cv2.FONT_HERSHEY_DUPLEX, .82, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(wall, f"source MOT: {args.mot.name}", (22, 70), cv2.FONT_HERSHEY_SIMPLEX, .62,
                    (190, 210, 235), 1, cv2.LINE_AA)
        for index, (_, card_filename) in enumerate(page_cards):
            card = cv2.imread(str(args.outdir / card_filename))
            if card is None:
                raise RuntimeError(f"cannot reload identity card: {card_filename}")
            row, column = divmod(index, wall_columns)
            thumb = fit_image(card, thumb_w - 8, thumb_h - 8)
            y, x = wall_header_h + row * thumb_h + 4, column * thumb_w + 4
            wall[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
        filename = ("all_global_ids_identity_wall.jpg" if len(cards) <= page_size
                    else f"all_global_ids_identity_wall_page_{page_index:02d}.jpg")
        write_image(args.outdir / filename, wall, 92)
        pages.append(filename)
    manifest_path = args.outdir / "identity_mosaics_manifest.json"
    manifest_path.write_text(json.dumps({"video": str(args.video), "fps": fps, "global_id_count": len(cards), "minimum_global_id_label": min(tracks) if tracks else None, "maximum_global_id_label": max_label if tracks else None, "labels_reindexed": False, "wall_pages": pages, "items": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"global_id_count": len(cards), "mosaics": len(cards), "wall_pages": [str(args.outdir / page) for page in pages]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
