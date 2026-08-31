from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mode_split.audit_mot import read_mot, sha256
from mode_split.open_set_team import (
    assign_open_set_team_modes,
    detect_persistent_team_switches,
    learn_open_set_team_modes,
)
from mode_split.quality_gate import quality_gated_torso_feature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--appearance-modes", type=int, default=6)
    parser.add_argument("--maximum-torso-occlusion", type=float, default=.18)
    parser.add_argument("--minimum-visible-fraction", type=float, default=.92)
    parser.add_argument("--minimum-informative-fraction", type=float, default=.18)
    parser.add_argument("--minimum-sharpness", type=float, default=8.0)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--minimum-confident", type=int, default=15)
    parser.add_argument("--purity", type=float, default=.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video, mot, output = args.video.resolve(), args.mot.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output must not exist: {output}")
    output.mkdir(parents=True)
    by_frame, gids = read_mot(mot)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    grouped_frames: dict[int, list[int]] = defaultdict(list)
    grouped_features: dict[int, list[np.ndarray]] = defaultdict(list)
    rejected = Counter()
    accepted = 0
    for frame_id in range(1, total + 1):
        ok, image = cap.read()
        if not ok:
            break
        rows = by_frame.get(frame_id, [])
        plain_boxes = [(x, y, w, h) for _, x, y, w, h in rows]
        for index, (gid, x, y, w, h) in enumerate(rows):
            others = plain_boxes[:index] + plain_boxes[index + 1:]
            feature, quality = quality_gated_torso_feature(
                image, (x, y, w, h), others,
                maximum_torso_occlusion=args.maximum_torso_occlusion,
                minimum_visible_fraction=args.minimum_visible_fraction,
                minimum_informative_fraction=args.minimum_informative_fraction,
                minimum_sharpness=args.minimum_sharpness,
            )
            if feature is None:
                rejected[quality.reason] += 1
                continue
            grouped_frames[gid].append(frame_id)
            grouped_features[gid].append(feature.astype(np.float16))
            accepted += 1
        if frame_id % 5000 == 0:
            print(json.dumps({"frame": frame_id, "accepted": accepted, "rejected": sum(rejected.values())}), flush=True)
    cap.release()

    flat_frames, flat_gids, flat_features = [], [], []
    for gid in sorted(grouped_frames):
        flat_frames.append(np.asarray(grouped_frames[gid], np.int32))
        flat_gids.append(np.full(len(grouped_frames[gid]), gid, np.int16))
        flat_features.append(np.asarray(grouped_features[gid], np.float16))
    frames = np.concatenate(flat_frames)
    feature_gids = np.concatenate(flat_gids)
    features = np.concatenate(flat_features).astype(np.float32)
    model = learn_open_set_team_modes(features, appearance_modes=args.appearance_modes)
    labels = assign_open_set_team_modes(features, model)
    np.savez_compressed(
        output / "quality_gated_team_observations.npz",
        frame=frames, gid=feature_gids, feature=features.astype(np.float16), team_mode=labels,
    )
    np.savez(output / "open_set_team_model.npz", centers=model.centers, radii=model.radii)

    switches = []
    for gid in sorted(grouped_frames):
        mask = feature_gids == gid
        found = detect_persistent_team_switches(
            frames[mask], labels[mask], window=args.window,
            minimum_confident=args.minimum_confident, purity=args.purity,
        )
        switches.extend({
            "global_id": gid, "cut_frame": item.frame,
            "before_mode": item.before_mode, "after_mode": item.after_mode,
            "before_support": item.before_support, "after_support": item.after_support,
        } for item in found)
    with (output / "team_switch_candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["global_id", "cut_frame", "before_mode", "after_mode", "before_support", "after_support"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(switches)

    gid11 = [row["cut_frame"] for row in switches if row["global_id"] == 11]
    report = {
        "schema_version": 1,
        "status": "experimental_candidates_require_video_review",
        "policy": "upper-torso jersey only; occlusion quality gate; two dominant modes plus open-set unknown",
        "sources": {"video": str(video), "mot": str(mot), "mot_sha256": sha256(mot)},
        "parameters": vars(args) | {"video": str(video), "mot": str(mot), "output": str(output)},
        "quality": {
            "mot_boxes": accepted + sum(rejected.values()), "accepted_torso_features": accepted,
            "rejected": dict(rejected), "accepted_fraction": accepted / max(1, accepted + sum(rejected.values())),
        },
        "model": {
            "appearance_modes": model.appearance_modes,
            "dominant_mode_sample_sizes": model.source_mode_sizes,
            "team_mode_separation": model.separation,
            "open_set_radii": model.radii.tolist(),
            "confident_team_fraction_of_accepted": float((labels >= 0).mean()),
        },
        "counts": {
            "input_global_ids": len(gids), "candidate_switches": len(switches),
            "ids_with_candidates": len({row["global_id"] for row in switches}),
        },
        "known_regression": {
            "global_id": 11, "expected_switch_window": [28168, 28184],
            "detected_cuts": gid11, "passed": any(28168 <= frame <= 28184 for frame in gid11),
        },
        "safety": {
            "mot_was_modified": False,
            "automatic_split_written": False,
            "reason": "candidate precision must be established by video review before changing identity IDs",
            "max_ids_merge_policy": "unchanged_upstream_result",
        },
    }
    (output / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
