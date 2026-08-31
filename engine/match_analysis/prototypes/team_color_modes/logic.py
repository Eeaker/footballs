"""PROTOTYPE: pure clustering logic for automatic jersey colour-mode discovery."""

from __future__ import annotations

import numpy as np


def cosine_distances(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = values / np.maximum(norms, 1e-12)
    return np.clip(1.0 - values @ values.T, 0.0, 2.0)


def average_linkage_labels(distances: np.ndarray, target_clusters: int) -> np.ndarray:
    clusters = [[index] for index in range(len(distances))]
    while len(clusters) > target_clusters:
        best = None
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                distance = float(distances[np.ix_(clusters[left], clusters[right])].mean())
                key = (distance, min(clusters[left]), min(clusters[right]))
                if best is None or key < best[0]:
                    best = (key, left, right)
        _, left, right = best
        clusters[left] = sorted(clusters[left] + clusters[right])
        del clusters[right]
    clusters.sort(key=lambda members: (-len(members), min(members)))
    labels = np.empty(len(distances), dtype=np.int32)
    for label, members in enumerate(clusters):
        labels[members] = label
    return labels


def silhouette_values(distances: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = np.zeros(len(labels), dtype=np.float64)
    unique = sorted(set(labels.tolist()))
    for index, label in enumerate(labels):
        same = np.flatnonzero(labels == label)
        same = same[same != index]
        if not len(same):
            continue
        own = float(distances[index, same].mean())
        other = min(
            float(distances[index, np.flatnonzero(labels == candidate)].mean())
            for candidate in unique if candidate != label
        )
        scale = max(own, other)
        result[index] = (other - own) / scale if scale > 1e-12 else 0.0
    return result


def discover_modes(features: np.ndarray, max_clusters: int = 6) -> dict:
    distances = cosine_distances(features)
    trials = []
    upper = min(max_clusters, len(features) - 1)
    for cluster_count in range(2, upper + 1):
        labels = average_linkage_labels(distances, cluster_count)
        silhouette = silhouette_values(distances, labels)
        sizes = [int((labels == label).sum()) for label in sorted(set(labels.tolist()))]
        singleton_fraction = sum(size == 1 for size in sizes) / len(labels)
        adjusted = float(silhouette.mean() - 0.10 * singleton_fraction)
        trials.append({
            "cluster_count": cluster_count,
            "labels": labels,
            "silhouette": silhouette,
            "mean_silhouette": float(silhouette.mean()),
            "adjusted_score": adjusted,
            "sizes": sizes,
        })
    best = max(trials, key=lambda item: (item["adjusted_score"], -item["cluster_count"]))
    return {"distances": distances, "trials": trials, "best": best}


def robust_outliers(distances: np.ndarray, labels: np.ndarray) -> dict:
    """Flag colour anomalies using Tukey fences with a between-team safety floor."""
    flagged = np.zeros(len(labels), dtype=bool)
    medoid_distances = np.zeros(len(labels), dtype=np.float64)
    thresholds = {}
    medoids = {}
    for label in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == label)
        within = distances[np.ix_(members, members)]
        medoid_local = int(np.argmin(within.mean(axis=1)))
        medoids[int(label)] = int(members[medoid_local])
    for label in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == label)
        medoid = medoids[int(label)]
        values = distances[members, medoid]
        q1, q3 = np.quantile(values, [.25, .75])
        other_medoids = [index for other, index in medoids.items() if other != int(label)]
        separation = min(float(distances[medoid, other]) for other in other_medoids)
        threshold = max(float(q3 + 1.5 * (q3 - q1)), 0.15 * separation)
        medoid_distances[members] = values
        flagged[members] = values > threshold
        thresholds[int(label)] = threshold
    return {
        "flagged": flagged,
        "medoid_distances": medoid_distances,
        "thresholds": thresholds,
        "medoids": medoids,
    }
