from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .io import MOTBox
from .tracking_adapter import aggregate_features, torso_feature


def assign_teams_kmeans(
    video: str | Path,
    mot_rows: list[MOTBox],
    n_clusters: int,
    samples_per_id: int = 12,
) -> tuple[dict[int, str], list[dict]]:
    """Aggregate several torso samples per global ID, then give each ID one equal K-means vote."""
    by_id: dict[int, list[MOTBox]] = defaultdict(list)
    for row in mot_rows:
        by_id[row.global_id].append(row)
    selected: dict[int, list[MOTBox]] = {}
    requests: dict[int, list[MOTBox]] = defaultdict(list)
    for identity, rows in by_id.items():
        rows.sort(key=lambda row: row.frame_proc)
        indices = np.linspace(0, len(rows) - 1, min(samples_per_id, len(rows)), dtype=int)
        selected[identity] = [rows[int(index)] for index in sorted(set(indices.tolist()))]
        for row in selected[identity]:
            requests[row.frame_proc].append(row)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开队色采样视频: {video}")
    features: dict[int, list[np.ndarray]] = defaultdict(list)
    for frame_index in sorted(requests):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        for box in requests[frame_index]:
            item = torso_feature(frame, (box.x, box.y, box.x + box.width, box.y + box.height))
            if item is not None:
                features[box.global_id].append(item[0])
    cap.release()
    identities = sorted(features)
    if len(identities) < n_clusters:
        raise ValueError(f"仅 {len(identities)} 个 ID 有有效球衣样本，无法聚为 {n_clusters} 类")
    prototypes = np.stack([aggregate_features(features[identity]) for identity in identities]).astype(np.float32)
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-5)
    _, labels, centers = cv2.kmeans(prototypes, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    # K-means cluster numbers are arbitrary. Sort centers lexicographically for reproducible team IDs.
    order = sorted(range(n_clusters), key=lambda index: tuple(np.round(centers[index], 6).tolist()))
    remap = {old: new for new, old in enumerate(order)}
    team_map: dict[int, str] = {}
    diagnostics: list[dict] = []
    for identity, label, vector in zip(identities, labels.flatten(), prototypes):
        distances = np.linalg.norm(centers - vector[None, :], axis=1)
        ranked = np.sort(distances)
        margin = float(ranked[1] - ranked[0]) if len(ranked) > 1 else 0.0
        team = f"team_{remap[int(label)]}"
        team_map[identity] = team
        diagnostics.append({
            "global_id": identity, "team_id": team, "samples": len(features[identity]),
            "nearest_center_distance": round(float(ranked[0]), 6),
            "center_margin": round(margin, 6), "assignment_method": "track_level_torso_hsv_kmeans",
        })
    for identity in sorted(by_id.keys() - team_map.keys()):
        team_map[identity] = "unassigned"
        diagnostics.append({
            "global_id": identity, "team_id": "unassigned", "samples": 0,
            "nearest_center_distance": "", "center_margin": "", "assignment_method": "no_valid_torso_sample",
        })
    return team_map, sorted(diagnostics, key=lambda row: row["global_id"])


def diagnostics_from_explicit_map(team_map: dict[int, str], identities: set[int]) -> list[dict]:
    return [{
        "global_id": identity, "team_id": team_map.get(identity, "unassigned"), "samples": "",
        "nearest_center_distance": "", "center_margin": "", "assignment_method": "explicit_frozen_map",
    } for identity in sorted(identities)]
