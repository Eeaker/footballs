"""PROTOTYPE QUESTION

Does automatic colour-mode discovery separate distinct jerseys better than forced K=2
on the existing U12 video, without any manually configured colour names?
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prototypes.team_color_modes.logic import discover_modes, robust_outliers
from analysis_lib.io import read_mot
from analysis_lib.tracking_adapter import (
    aggregate_features, representative_hsv, suggested_color_name, torso_feature,
)


def collect(video: Path, mot: Path, samples_per_id: int):
    _, rows = read_mot(mot)
    by_id = defaultdict(list)
    for row in rows:
        by_id[row.global_id].append(row)
    requests = defaultdict(list)
    for identity, identity_rows in by_id.items():
        identity_rows.sort(key=lambda row: row.frame_proc)
        indices = np.linspace(0, len(identity_rows) - 1, min(samples_per_id, len(identity_rows)), dtype=int)
        for index in sorted(set(indices.tolist())):
            requests[identity_rows[index].frame_proc].append(identity_rows[index])
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    features, crops = defaultdict(list), defaultdict(list)
    for frame_index in sorted(requests):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        for box in requests[frame_index]:
            item = torso_feature(frame, (box.x, box.y, box.x + box.width, box.y + box.height))
            if item is not None:
                features[box.global_id].append(item[0])
                crops[box.global_id].append(item[1])
    cap.release()
    identities = sorted(features)
    prototypes = np.stack([aggregate_features(features[identity]) for identity in identities]).astype(np.float32)
    return identities, prototypes, features, crops


def kmeans2(features: np.ndarray):
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-5)
    _, labels, centers = cv2.kmeans(features, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    order = sorted(range(2), key=lambda index: tuple(np.round(centers[index], 6).tolist()))
    remap = {old: new for new, old in enumerate(order)}
    return np.asarray([remap[int(label)] for label in labels], dtype=np.int32)


def load_evaluation(path: Path | None):
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(item["global_id"]): f"{item['team']}_{item['final_number']}"
        for item in data.get("eligible_confirmed", [])
        if item.get("team") and item.get("final_number") not in (None, "")
    }


def contact_sheets(output: Path, identities, labels, crops, colour_names):
    for label in sorted(set(labels.tolist())):
        members = [identity for identity, item_label in zip(identities, labels) if item_label == label]
        tiles = []
        for identity in members:
            crop = crops[identity][len(crops[identity]) // 2]
            tile = np.full((130, 120, 3), 245, dtype=np.uint8)
            scale = min(110 / crop.shape[1], 92 / crop.shape[0])
            resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
            x = (120 - resized.shape[1]) // 2
            tile[4:4 + resized.shape[0], x:x + resized.shape[1]] = resized
            cv2.putText(tile, f"ID {identity}", (5, 108), cv2.FONT_HERSHEY_SIMPLEX, .42, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(tile, colour_names[identity], (5, 124), cv2.FONT_HERSHEY_SIMPLEX, .4, (50, 50, 50), 1, cv2.LINE_AA)
            tiles.append(tile)
        columns = 6
        rows = (len(tiles) + columns - 1) // columns
        sheet = np.full((rows * 130, columns * 120, 3), 255, dtype=np.uint8)
        for index, tile in enumerate(tiles):
            y, x = divmod(index, columns)
            sheet[y * 130:(y + 1) * 130, x * 120:(x + 1) * 120] = tile
        cv2.imwrite(str(output / f"auto_cluster_{label}.jpg"), sheet)


def main():
    parser = argparse.ArgumentParser(description="PROTOTYPE: automatic jersey colour modes versus fixed K=2")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, help="post-hoc confirmed identities; never used for clustering")
    parser.add_argument("--samples-per-id", type=int, default=12)
    parser.add_argument("--max-clusters", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"prototype output already exists: {args.output}")
    args.output.mkdir(parents=True)
    identities, features, sampled, crops = collect(args.video, args.mot, args.samples_per_id)
    result = discover_modes(features, args.max_clusters)
    best = result["best"]
    labels = best["labels"]
    anomaly = robust_outliers(result["distances"], labels)
    k3_trial = next((trial for trial in result["trials"] if trial["cluster_count"] == 3), None)
    baseline = kmeans2(features)
    evaluation = load_evaluation(args.evaluation)
    sizes = {label: int((labels == label).sum()) for label in sorted(set(labels.tolist()))}
    dominant = set(sorted(sizes, key=lambda label: (-sizes[label], label))[:2])
    colour_names = {}
    rows = []
    for index, identity in enumerate(identities):
        hsv = representative_hsv(crops[identity])
        colour = suggested_color_name(hsv)
        colour_names[identity] = colour
        label = int(labels[index])
        confidence = float(best["silhouette"][index])
        is_outlier = bool(anomaly["flagged"][index])
        rows.append({
            "global_id": identity,
            "samples": len(sampled[identity]),
            "representative_h": round(float(hsv[0]), 3),
            "representative_s": round(float(hsv[1]), 3),
            "representative_v": round(float(hsv[2]), 3),
            "suggested_colour": colour,
            "fixed_k2_cluster": f"team_{int(baseline[index])}",
            "auto_colour_cluster": f"colour_{label}",
            "alternative_k3_cluster": f"colour_{int(k3_trial['labels'][index])}" if k3_trial else "",
            "auto_cluster_size": sizes[label],
            "auto_role": "special_colour_or_outlier" if is_outlier else f"team_candidate_{label}",
            "cluster_medoid_distance": round(float(anomaly["medoid_distances"][index]), 6),
            "cluster_outlier_threshold": round(float(anomaly["thresholds"][label]), 6),
            "robust_colour_outlier": int(is_outlier),
            "silhouette": round(confidence, 6),
            "confidence": "low" if confidence < .08 else "medium" if confidence < .25 else "high",
            "confirmed_identity_for_evaluation": evaluation.get(identity, ""),
        })
    with (args.output / "cluster_assignments.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (args.output / "experimental_team_map.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["global_id", "team_id", "assignment_method", "colour_cluster", "robust_colour_outlier"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "global_id": row["global_id"],
                "team_id": "unassigned" if row["robust_colour_outlier"] else row["fixed_k2_cluster"],
                "assignment_method": "prototype_k2_with_automatic_outlier_rejection",
                "colour_cluster": row["auto_colour_cluster"],
                "robust_colour_outlier": row["robust_colour_outlier"],
            })
    trials = [{key: value for key, value in trial.items() if key not in {"labels", "silhouette"}} for trial in result["trials"]]
    report = {
        "prototype": True,
        "question": "Does automatic colour-mode discovery avoid forcing every jersey into K=2?",
        "input": {"video": str(args.video.resolve()), "mot": str(args.mot.resolve())},
        "identities": len(identities),
        "selected_cluster_count": int(best["cluster_count"]),
        "selected_cluster_sizes": best["sizes"],
        "mean_silhouette": best["mean_silhouette"],
        "adjusted_score": best["adjusted_score"],
        "robust_colour_outlier_ids": [
            int(identity) for identity, flagged in zip(identities, anomaly["flagged"]) if flagged
        ],
        "trials": trials,
        "evaluation_is_posthoc_only": True,
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheets(args.output, identities, labels, crops, colour_names)
    lines = [
        "# PROTOTYPE - 自动球衣颜色模式实验", "",
        f"- 轨迹 ID：{len(identities)}", f"- 自动选择簇数：{best['cluster_count']}",
        f"- 簇大小：{best['sizes']}", f"- 平均轮廓系数：{best['mean_silhouette']:.4f}", "",
        "确认号码仅用于输出后对照，没有进入聚类。详见 `cluster_assignments.csv` 和各簇 contact sheet。",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
