from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def read_tracks(path: Path):
    tracks = defaultdict(dict)
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            v = line.rstrip().split(",")
            if len(v) >= 6:
                frame, gid = int(float(v[0])), int(float(v[1]))
                tracks[gid][frame] = tuple(map(float, v[2:6]))
    return tracks


def nearest(mapping: dict, target: int):
    if target in mapping:
        return target, mapping[target]
    candidates = [frame for frame in mapping if abs(frame - target) <= 8]
    if not candidates:
        return None
    frame = min(candidates, key=lambda item: abs(item - target))
    return frame, mapping[frame]


def frame_tile(cap, frame_id: int, box, label: str):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
    ok, frame = cap.read()
    canvas = np.full((260, 420, 3), 245, np.uint8)
    if not ok:
        return canvas
    x, y, w, h = box
    pad_x, pad_y = 1.8 * w, .65 * h
    x1, y1 = max(0, int(x - pad_x)), max(0, int(y - pad_y))
    x2, y2 = min(frame.shape[1], int(x + w + pad_x)), min(frame.shape[0], int(y + h + pad_y))
    crop = frame[y1:y2, x1:x2].copy()
    cv2.rectangle(crop, (int(x - x1), int(y - y1)), (int(x + w - x1), int(y + h - y1)), (0, 255, 255), 3)
    scale = min(420 / max(1, crop.shape[1]), 230 / max(1, crop.shape[0]))
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
    ox, oy = (420 - crop.shape[1]) // 2, (230 - crop.shape[0]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    cv2.putText(canvas, label, (6, 252), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--mot", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pair", action="append", required=True, help="global_id:cut_frame")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    tracks = read_tracks(args.mot)
    cap = cv2.VideoCapture(str(args.video))
    offsets = [-60, -45, -30, -15, -5, 0, 5, 15, 30, 45, 60, 90]
    for value in args.pair:
        gid, cut = map(int, value.split(":"))
        tiles = []
        for offset in offsets:
            item = nearest(tracks[gid], cut + offset)
            if item is None:
                tiles.append(np.full((260, 420, 3), 230, np.uint8))
            else:
                frame_id, box = item
                tiles.append(frame_tile(cap, frame_id, box, f"gid={gid} cut={cut} f={frame_id} dt={frame_id-cut}"))
        sheet = np.full((3 * 260, 4 * 420, 3), 255, np.uint8)
        for index, tile in enumerate(tiles):
            row, column = divmod(index, 4)
            sheet[row * 260:(row + 1) * 260, column * 420:(column + 1) * 420] = tile
        cv2.imwrite(str(args.output / f"gid_{gid}_cut_{cut}_storyboard.jpg"), sheet)
    cap.release()


if __name__ == "__main__":
    main()
