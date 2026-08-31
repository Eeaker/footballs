from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import cv2
import numpy as np


def read_tracks(path: Path):
    tracks = defaultdict(dict)
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            v = line.rstrip().split(",")
            if len(v) >= 6:
                f, g = int(float(v[0])), int(float(v[1]))
                tracks[g][f] = tuple(map(float, v[2:6]))
    return tracks


def nearest(mapping: dict, target: int, direction: int):
    candidates = [frame for frame in mapping if (frame < target if direction < 0 else frame >= target)]
    if not candidates:
        return None
    frame = max(candidates) if direction < 0 else min(candidates)
    return (frame, mapping[frame]) if abs(frame - target) <= 60 else None


def tile(cap, frame_id: int, box, text: str):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
    ok, frame = cap.read()
    canvas = np.full((220, 320, 3), 245, np.uint8)
    if not ok:
        return canvas
    x, y, w, h = box
    pad_x, pad_y = .7 * w, .25 * h
    x1, y1 = max(0, int(x - pad_x)), max(0, int(y - pad_y))
    x2, y2 = min(frame.shape[1], int(x + w + pad_x)), min(frame.shape[0], int(y + h + pad_y))
    crop = frame[y1:y2, x1:x2].copy()
    cv2.rectangle(crop, (int(x - x1), int(y - y1)), (int(x + w - x1), int(y + h - y1)), (0, 255, 255), 2)
    scale = min(320 / max(1, crop.shape[1]), 190 / max(1, crop.shape[0]))
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
    ox, oy = (320 - crop.shape[1]) // 2, (190 - crop.shape[0]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    cv2.putText(canvas, text, (5, 212), cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--mot", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--observations", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--samples-per-id", type=int, default=1)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    tracks = read_tracks(args.mot)
    rows = list(csv.DictReader(args.candidates.open(encoding="utf-8-sig")))
    observations = None
    if args.observations:
        data = np.load(args.observations)
        observations = (data["frame"], data["gid"], data["team_mode"])

    # Deterministic breadth sample across IDs and time, plus the fixed regression.
    selected, seen = [], set()
    known = next((row for row in rows if int(row["global_id"]) == 11 and 28168 <= int(row["cut_frame"]) <= 28184), None)
    if known:
        selected.append(known); seen.add(11)
    by_gid = defaultdict(list)
    for row in rows:
        by_gid[int(row["global_id"])].append(row)
    for gid in sorted(by_gid):
        candidates = by_gid[gid]
        indices = np.linspace(0, len(candidates) - 1, min(args.samples_per_id, len(candidates)), dtype=int)
        for index in sorted(set(indices.tolist())):
            row = candidates[index]
            key = (gid, int(row["cut_frame"]))
            if not any((int(item["global_id"]), int(item["cut_frame"])) == key for item in selected):
                selected.append(row)
    cap = cv2.VideoCapture(str(args.video))
    pair_tiles = []
    for row in selected:
        gid, cut = int(row["global_id"]), int(row["cut_frame"])
        before, after = None, None
        if observations is not None:
            obs_frames, obs_gids, obs_labels = observations
            mask = obs_gids == gid
            local_frames, local_labels = obs_frames[mask], obs_labels[mask]
            # Global-ID merge errors often join tracklets separated by many
            # seconds.  Compare the last confident observation of the previous
            # mode with the first confident observations of the next mode;
            # restricting this to a short cut-centred clip hides that evidence.
            before_frames = local_frames[(local_frames < cut)
                                         & (local_labels == int(row["before_mode"]))]
            after_frames = local_frames[(local_frames >= cut)
                                        & (local_labels == int(row["after_mode"]))]
            if len(before_frames) and len(after_frames):
                before_frame = int(before_frames[max(0, len(before_frames) - 5)])
                after_frame = int(after_frames[min(4, len(after_frames) - 1)])
                if before_frame in tracks[gid] and after_frame in tracks[gid]:
                    before = before_frame, tracks[gid][before_frame]
                    after = after_frame, tracks[gid][after_frame]
        if before is None or after is None:
            before, after = nearest(tracks[gid], cut, -1), nearest(tracks[gid], cut, 1)
        if not before or not after:
            continue
        support = f"support={row['before_support']}/{row['after_support']}"
        pair_tiles.append((
            tile(cap, before[0], before[1], f"gid={gid} BEFORE f={before[0]} m={row['before_mode']} {support}"),
            tile(cap, after[0], after[1], f"gid={gid} AFTER f={after[0]} m={row['after_mode']} cut={cut}"),
        ))
    cap.release()
    per_page = 6
    for page_start in range(0, len(pair_tiles), per_page):
        page = np.full((per_page * 220, 640, 3), 255, np.uint8)
        for index, (left, right) in enumerate(pair_tiles[page_start:page_start + per_page]):
            y = index * 220
            page[y:y + 220, :320] = left
            page[y:y + 220, 320:] = right
        cv2.imwrite(str(args.output / f"candidate_contact_sheet_{page_start // per_page + 1:02d}.jpg"), page)


if __name__ == "__main__":
    main()
