from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mode_split.audit_mot import box_feature, read_mot, sha256, write_split_mot
from mode_split.local_change import detect_local_feature_transitions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-gids", type=int, nargs="*")
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--absolute-distance", type=float, default=.35)
    parser.add_argument("--variability-ratio", type=float, default=2.2)
    parser.add_argument("--return-horizon-frames", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video, mot, output = args.video.resolve(), args.mot.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output must not exist: {output}")
    if args.frame_start < 1 or (args.frame_end is not None and args.frame_end < args.frame_start):
        raise ValueError("invalid frame range")
    output.mkdir(parents=True)
    selected = set(args.only_gids) if args.only_gids else None
    by_frame, gids = read_mot(
        mot, selected, frame_start=args.frame_start, frame_end=args.frame_end,
    )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(total_video_frames, args.frame_end or total_video_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_start - 1)
    grouped_frames: dict[int, list[int]] = defaultdict(list)
    grouped_features: dict[int, list[np.ndarray]] = defaultdict(list)
    invalid_boxes = 0
    for frame_id in range(args.frame_start, end + 1):
        ok, image = cap.read()
        if not ok:
            break
        for gid, x, y, w, h in by_frame.get(frame_id, []):
            feature = box_feature(image, (x, y, w, h))
            if feature is None:
                invalid_boxes += 1
                continue
            grouped_frames[gid].append(frame_id)
            grouped_features[gid].append(feature.astype(np.float16))
        if frame_id % 5000 == 0:
            observed = sum(len(rows) for rows in grouped_frames.values())
            print(json.dumps({"frame": frame_id, "observed_boxes": observed}), flush=True)
    cap.release()

    flat_frames, flat_gids, flat_features = [], [], []
    transition_rows = []
    cuts_by_gid: dict[int, list[int]] = {}
    for gid in sorted(grouped_frames):
        frames = np.asarray(grouped_frames[gid], np.int32)
        features = np.asarray(grouped_features[gid], np.float32)
        flat_frames.append(frames)
        flat_gids.append(np.full(len(frames), gid, np.int16))
        flat_features.append(features.astype(np.float16))
        cuts = detect_local_feature_transitions(
            frames, features, window=args.window,
            absolute_distance=args.absolute_distance,
            variability_ratio=args.variability_ratio,
            return_horizon_frames=args.return_horizon_frames,
        )
        if cuts:
            cuts_by_gid[gid] = [row.frame for row in cuts]
        for row in cuts:
            transition_rows.append({
                "global_id": gid, "cut_frame": row.frame,
                "feature_distance": round(row.feature_distance, 8),
                "adaptive_ratio": round(row.adaptive_ratio, 8),
                "left_variability": round(row.left_variability, 8),
                "right_variability": round(row.right_variability, 8),
            })
    if flat_frames:
        np.savez_compressed(
            output / "per_box_local_features.npz",
            frame=np.concatenate(flat_frames), gid=np.concatenate(flat_gids),
            feature=np.concatenate(flat_features),
        )
    fields = ["global_id", "cut_frame", "feature_distance", "adaptive_ratio",
              "left_variability", "right_variability"]
    with (output / "local_transitions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(transition_rows)

    mapping = []
    if selected is None and args.frame_start == 1 and args.frame_end is None:
        mapping = write_split_mot(mot, output / "tracking_mot_local_change_split.txt", cuts_by_gid)
        with (output / "segment_map.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mapping[0]))
            writer.writeheader()
            writer.writerows(mapping)

    gid11_cuts = cuts_by_gid.get(11, [])
    report = {
        "schema_version": 1,
        "status": "experimental_not_for_delivery",
        "policy": "track-local per-detection-box colour change; no global K; no colour/team input",
        "sources": {"video": str(video), "mot": str(mot), "mot_sha256": sha256(mot)},
        "parameters": {
            "window": args.window, "absolute_distance": args.absolute_distance,
            "variability_ratio": args.variability_ratio,
            "return_horizon_frames": args.return_horizon_frames,
            "frame_start": args.frame_start, "frame_end": args.frame_end,
            "max_ids_merge_policy": "unchanged_upstream_result",
        },
        "counts": {
            "input_global_ids": len(gids),
            "valid_box_features": sum(len(rows) for rows in grouped_frames.values()),
            "invalid_boxes": invalid_boxes,
            "detected_transitions": len(transition_rows),
            "ids_with_transitions": len(cuts_by_gid),
            "output_segments": len(mapping) if mapping else None,
        },
        "known_regression": {
            "global_id": 11, "expected_switch_window": [28168, 28184],
            "detected_cuts": gid11_cuts,
            "passed": any(28168 <= frame <= 28184 for frame in gid11_cuts),
        },
        "limitations": [
            "Detects sustained within-track appearance change, not same-colour same-team identity switches.",
            "A real kit change and a persistent contaminated crop can look alike and require video review.",
            "The upstream greedy max_ids=10 merge result is deliberately unchanged for this A/B.",
        ],
    }
    (output / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report["counts"] | {"known_regression": report["known_regression"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
