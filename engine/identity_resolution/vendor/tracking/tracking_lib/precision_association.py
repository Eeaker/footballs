from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import copy
import math

import cv2
import numpy as np

from tracking_lib.team_features import aggregate_features


@dataclass(frozen=True)
class AssociationConfig:
    min_track_frames: int = 1
    min_presence_ratio: float = 0.0
    short_gap_seconds: float = 5.0
    long_gap_seconds: float = 30.0
    maximum_merge_gap_seconds: float = 30.0
    maximum_speed_mps: float = 11.0
    metric_uncertainty_m: float = 2.0
    kit_confidence_min: float = .72
    kit_hue_conflict_degrees: float = 25.0
    kit_hue_same_degrees: float = 14.0
    colour_min: float = .15
    appearance_weight: float = .6
    colour_weight: float = .4
    short_reid_min: float = .70
    medium_reid_min: float = .78
    long_reid_min: float = .85
    short_score_min: float = .75
    medium_score_min: float = .82
    long_score_min: float = .88
    reciprocal_top_k: int = 3
    team_clusters: int = 3
    team_min_train_frames: int = 60
    team_min_train_samples: int = 4
    team_max_distance: float = .35
    team_min_margin: float = .08


@dataclass(frozen=True)
class KitDescriptor:
    family: str
    hue_degrees: float | None
    confidence: float
    samples: int


@dataclass(frozen=True)
class PairDecision:
    left: int
    right: int
    accepted: bool
    reason: str
    gap_seconds: float | None = None
    metric_distance_m: float | None = None
    metric_speed_mps: float | None = None
    embedding_similarity: float | None = None
    colour_similarity: float | None = None
    score: float | None = None
    tier: str | None = None


@dataclass(frozen=True)
class AssociationResult:
    local_to_global: dict[int, int]
    audit: dict


@dataclass(frozen=True)
class TrackletSplitResult:
    tracklets: dict
    metric_positions: dict[int, list[tuple[int, float, float]]]
    frame_id_remap: dict[tuple[int, int], int]
    report: dict


def _normalised_feature(tracklet: dict) -> tuple[np.ndarray, np.ndarray]:
    embedding = np.asarray(tracklet.get("emb_sum", []), np.float32)
    if int(tracklet.get("emb_cnt", 0)) > 0:
        embedding = embedding / float(tracklet["emb_cnt"])
    embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-9)
    colour = np.asarray(tracklet.get("col_sum", []), np.float32)
    if int(tracklet.get("col_cnt", 0)) > 0:
        colour = colour / float(tracklet["col_cnt"])
    return embedding, colour


def describe_kit(tracklet: dict) -> KitDescriptor:
    samples = [np.asarray(row[1], np.float32) for row in tracklet.get("team_feature_samples", [])]
    samples = [row for row in samples if row.size >= 6 and np.all(np.isfinite(row))]
    if len(samples) < 3:
        return KitDescriptor("unknown", None, 0.0, len(samples))

    white_votes, colour_votes, angles = [], [], []
    for feature in samples:
        white_strength = max(float(feature[-2]), 0.0)
        colour_strength = max(float(feature[-1]), 0.0)
        total = white_strength + colour_strength + 1e-9
        white_votes.append(white_strength / total)
        colour_votes.append(colour_strength / total)
        if colour_strength > white_strength:
            angles.append(math.atan2(float(feature[-5]), float(feature[-6])))

    sample_factor = min(1.0, len(samples) / 4.0)
    white_score = float(np.median(white_votes))
    colour_score = float(np.median(colour_votes))
    if white_score >= .70:
        return KitDescriptor("white", None, white_score * sample_factor, len(samples))
    if colour_score < .70 or not angles:
        return KitDescriptor("unknown", None, max(white_score, colour_score) * .5, len(samples))

    cosine = float(np.mean(np.cos(angles)))
    sine = float(np.mean(np.sin(angles)))
    concentration = math.hypot(cosine, sine)
    hue = math.degrees(math.atan2(sine, cosine)) % 360.0
    return KitDescriptor("colour", hue, colour_score * concentration * sample_factor, len(samples))


def _circular_degrees(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _single_sample_kit(feature: np.ndarray) -> KitDescriptor:
    feature = np.asarray(feature, np.float32)
    if feature.size < 6 or not np.all(np.isfinite(feature)):
        return KitDescriptor("unknown", None, 0.0, 0)
    white_strength, colour_strength = max(float(feature[-2]), 0.0), max(float(feature[-1]), 0.0)
    total = white_strength + colour_strength + 1e-9
    white_score, colour_score = white_strength / total, colour_strength / total
    if white_score >= .70:
        return KitDescriptor("white", None, white_score, 1)
    if colour_score < .70:
        return KitDescriptor("unknown", None, max(white_score, colour_score) * .5, 1)
    hue = math.degrees(math.atan2(float(feature[-5]), float(feature[-6]))) % 360.0
    return KitDescriptor("colour", hue, colour_score, 1)


def _single_sample_colour_observation(
    feature: np.ndarray, *, minimum_colour_fraction: float = .40,
    minimum_hue_strength: float = .12,
) -> KitDescriptor:
    """Return a permissive hue observation for local-switch detection only.

    Dark/blue jerseys often contain enough bright neutral pixels for the strict
    kit classifier to abstain.  Their circular hue signal is still reliable
    enough to propose a change point when an independent ReID change agrees.
    This helper must not be used as the pairwise hard team gate by itself.
    """
    feature = np.asarray(feature, np.float32)
    if feature.size < 6 or not np.all(np.isfinite(feature)):
        return KitDescriptor("unknown", None, 0.0, 0)
    white_strength = max(float(feature[-2]), 0.0)
    colour_strength = max(float(feature[-1]), 0.0)
    colour_fraction = colour_strength / (white_strength + colour_strength + 1e-9)
    hue_cos, hue_sin = float(feature[-6]), float(feature[-5])
    hue_strength = math.hypot(hue_cos, hue_sin)
    if colour_fraction < minimum_colour_fraction or hue_strength < minimum_hue_strength:
        return KitDescriptor("unknown", None, 0.0, 1)
    hue = math.degrees(math.atan2(hue_sin, hue_cos)) % 360.0
    return KitDescriptor("colour", hue, colour_fraction, 1)


def describe_stable_colour(tracklet: dict) -> KitDescriptor:
    """Describe a stable chromatic kit even when the strict family vote abstains."""
    observations = [
        _single_sample_colour_observation(row[1])
        for row in tracklet.get("team_feature_samples", ())
    ]
    angles = [
        math.radians(float(row.hue_degrees))
        for row in observations if row.family == "colour"
    ]
    if len(angles) < 3:
        return KitDescriptor("unknown", None, 0.0, len(angles))
    cosine = float(np.mean(np.cos(angles)))
    sine = float(np.mean(np.sin(angles)))
    concentration = math.hypot(cosine, sine)
    hue = math.degrees(math.atan2(sine, cosine)) % 360.0
    sample_factor = min(1.0, len(angles) / 4.0)
    return KitDescriptor("colour", hue, concentration * sample_factor, len(angles))


def find_kit_outlier_frames(tracklet: dict, config: AssociationConfig) -> set[int]:
    """Find short windows where a local ID's observed kit contradicts its dominant kit."""
    dominant = describe_kit(tracklet)
    if dominant.confidence < config.kit_confidence_min or dominant.family == "unknown":
        return set()
    samples = sorted(tracklet.get("team_feature_samples", ()), key=lambda row: int(row[0]))
    if len(samples) < 3:
        return set()
    sample_frames = np.asarray([int(row[0]) for row in samples], np.int64)
    steps = np.diff(sample_frames)
    radius = max(1, int(round(float(np.median(steps[steps > 0])) / 2.0))) if np.any(steps > 0) else 1
    conflicts = []
    for frame, feature in samples:
        observed = _single_sample_kit(feature)
        if observed.confidence < config.kit_confidence_min or observed.family == "unknown":
            continue
        if observed.family != dominant.family:
            conflicts.append(int(frame))
        elif (observed.family == "colour" and _circular_degrees(
                float(observed.hue_degrees), float(dominant.hue_degrees)
        ) >= config.kit_hue_conflict_degrees):
            conflicts.append(int(frame))
    if not conflicts:
        return set()
    frames = set(map(int, tracklet.get("frames", ())))
    return {
        frame for frame in frames
        if any(abs(frame - conflict) <= radius for conflict in conflicts)
    }


def _window_colour_hue(rows) -> float | None:
    angles = [math.radians(float(row[2].hue_degrees)) for row in rows if row[2].family == "colour"]
    if len(angles) != len(rows) or not angles:
        return None
    return math.degrees(math.atan2(
        float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles)))
    )) % 360.0


def _rebuild_tracklet_segment(tracklet: dict, frames: set[int]) -> dict:
    segment = copy.deepcopy(tracklet)
    segment["frames"] = set(frames)
    segment["first"], segment["last"] = min(frames), max(frames)
    segment["appearance_samples"] = [
        row for row in tracklet.get("appearance_samples", ()) if int(row[0]) in frames
    ]
    segment["team_feature_samples"] = [
        row for row in tracklet.get("team_feature_samples", ()) if int(row[0]) in frames
    ]
    if segment["appearance_samples"]:
        segment["emb_sum"] = np.sum([row[1] for row in segment["appearance_samples"]], axis=0).astype(np.float32)
        segment["col_sum"] = np.sum([row[2] for row in segment["appearance_samples"]], axis=0).astype(np.float32)
        segment["emb_cnt"] = segment["col_cnt"] = len(segment["appearance_samples"])
    else:
        segment["emb_sum"] = np.zeros_like(np.asarray(tracklet.get("emb_sum", []), np.float32))
        segment["col_sum"] = np.zeros_like(np.asarray(tracklet.get("col_sum", []), np.float32))
        segment["emb_cnt"] = segment["col_cnt"] = 0
    segment["team_feature"] = aggregate_features([row[1] for row in segment["team_feature_samples"]])
    return segment


def quarantine_tracklet_frames(
    tracklets: dict,
    metric_positions: dict[int, list[tuple[int, float, float]]],
    rejected_frames: dict[int, set[int]],
    *, component_gap_frames: int = 3,
) -> TrackletSplitResult:
    """Preserve suspicious local frames as non-mergeable singleton segments.

    Purity filtering used to delete these detections.  Quarantine keeps recall:
    the clean portion may still enter association, while each suspicious
    temporal component receives its own ID and is never offered to ReID.
    """
    next_id = max(map(int, tracklets), default=0) + 1
    output, output_positions, remap, rows = {}, {}, {}, []
    for track_id in sorted(tracklets):
        tracklet = tracklets[track_id]
        all_frames = set(map(int, tracklet.get("frames", ())))
        rejected = all_frames & set(map(int, rejected_frames.get(track_id, ())))
        accepted = all_frames - rejected
        if accepted:
            output[track_id] = _rebuild_tracklet_segment(tracklet, accepted)
            output_positions[track_id] = [
                row for row in metric_positions.get(track_id, ()) if int(row[0]) in accepted
            ]
            for frame in accepted:
                remap[(int(track_id), frame)] = int(track_id)
        if not rejected:
            continue
        components = []
        for frame in sorted(rejected):
            if not components or frame - components[-1][-1] > component_gap_frames:
                components.append([frame])
            else:
                components[-1].append(frame)
        segment_ids = []
        for component in components:
            segment_id = next_id
            next_id += 1
            segment_ids.append(segment_id)
            frame_set = set(component)
            segment = _rebuild_tracklet_segment(tracklet, frame_set)
            segment["association_quarantine"] = True
            segment["source_track_id"] = int(track_id)
            output[segment_id] = segment
            output_positions[segment_id] = [
                row for row in metric_positions.get(track_id, ()) if int(row[0]) in frame_set
            ]
            for frame in frame_set:
                remap[(int(track_id), frame)] = segment_id
        rows.append({
            "original_track_id": int(track_id),
            "quarantine_track_ids": segment_ids,
            "quarantined_frames": len(rejected),
        })
    return TrackletSplitResult(
        output, output_positions, remap,
        {"method": "non_destructive_local_outlier_quarantine",
         "affected_tracklets": len(rows),
         "quarantined_detections": sum(row["quarantined_frames"] for row in rows),
         "quarantine_segments": sum(len(row["quarantine_track_ids"]) for row in rows),
         "tracks": rows},
    )


def split_tracklets_on_persistent_kit_change(
    tracklets: dict, metric_positions: dict[int, list[tuple[int, float, float]]],
    config: AssociationConfig, *, window_samples: int = 4,
    minimum_hue_shift_degrees: float = 60.0,
    maximum_cross_window_reid_similarity: float = .85,
    minimum_segment_frames: int = 30, maximum_splits_per_track: int = 3,
    minimum_split_colour_fraction: float = .40,
    minimum_split_hue_strength: float = .12,
) -> TrackletSplitResult:
    """Split long A->B local-ID switches before global association.

    This ports GTA-Link's "split before connect" idea while making the split
    temporal: both a persistent kit-hue change and an independent embedding
    change are required. Non-contiguous clustering labels are never emitted.
    """
    next_id = max(map(int, tracklets), default=0) + 1
    output, output_positions, remap, split_rows = {}, {}, {}, []
    for track_id in sorted(tracklets):
        tracklet = tracklets[track_id]
        team_by_frame = {int(row[0]): row[1] for row in tracklet.get("team_feature_samples", ())}
        rows = []
        for frame, embedding, _colour in tracklet.get("appearance_samples", ()):
            descriptor = _single_sample_colour_observation(
                team_by_frame.get(int(frame), np.asarray([])),
                minimum_colour_fraction=minimum_split_colour_fraction,
                minimum_hue_strength=minimum_split_hue_strength,
            )
            vector = np.asarray(embedding, np.float32)
            norm = float(np.linalg.norm(vector))
            if descriptor.family == "colour" and norm > 1e-9:
                rows.append((int(frame), vector / norm, descriptor))
        candidates, width = [], max(2, int(window_samples))
        for index in range(width, len(rows) - width + 1):
            left, right = rows[index - width:index], rows[index:index + width]
            left_hue, right_hue = _window_colour_hue(left), _window_colour_hue(right)
            if left_hue is None or right_hue is None:
                continue
            hue_shift = _circular_degrees(left_hue, right_hue)
            left_embedding = np.mean([row[1] for row in left], axis=0)
            right_embedding = np.mean([row[1] for row in right], axis=0)
            similarity = float(np.dot(left_embedding, right_embedding) / max(
                float(np.linalg.norm(left_embedding) * np.linalg.norm(right_embedding)), 1e-9
            ))
            if hue_shift >= minimum_hue_shift_degrees and similarity < maximum_cross_window_reid_similarity:
                candidates.append({
                    "index": index, "left_frame": left[-1][0], "right_frame": right[0][0],
                    "hue_shift_degrees": hue_shift, "reid_similarity": similarity,
                    "strength": hue_shift + 100.0 * (maximum_cross_window_reid_similarity - similarity),
                })
        bands = []
        for candidate in candidates:
            if not bands or candidate["index"] > bands[-1][-1]["index"] + 1:
                bands.append([candidate])
            else:
                bands[-1].append(candidate)
        selected = sorted(
            (max(band, key=lambda row: row["strength"]) for band in bands),
            key=lambda row: row["strength"], reverse=True,
        )[:maximum_splits_per_track]
        selected.sort(key=lambda row: row["left_frame"])
        cuts, track_frames = [], sorted(map(int, tracklet.get("frames", ())))
        for candidate in selected:
            cut = (candidate["left_frame"] + candidate["right_frame"]) // 2
            trial = sorted(cuts + [cut])
            bounds = [track_frames[0] - 1, *trial, track_frames[-1]] if track_frames else []
            if bounds and all(bounds[i + 1] - bounds[i] >= minimum_segment_frames for i in range(len(bounds) - 1)):
                cuts = trial
        if not cuts:
            output[track_id] = copy.deepcopy(tracklet)
            output_positions[track_id] = list(metric_positions.get(track_id, ()))
            for frame in tracklet.get("frames", ()):
                remap[(int(track_id), int(frame))] = int(track_id)
            continue
        segment_ids = [int(track_id)] + list(range(next_id, next_id + len(cuts)))
        next_id += len(cuts)
        frame_groups = [[] for _ in segment_ids]
        for frame in track_frames:
            segment_index = int(np.searchsorted(cuts, frame, side="left"))
            frame_groups[segment_index].append(frame)
            remap[(int(track_id), frame)] = segment_ids[segment_index]
        for segment_id, frames in zip(segment_ids, frame_groups):
            if not frames:
                continue
            frame_set = set(frames)
            output[segment_id] = _rebuild_tracklet_segment(tracklet, frame_set)
            output_positions[segment_id] = [
                row for row in metric_positions.get(track_id, ()) if int(row[0]) in frame_set
            ]
        split_rows.append({
            "original_track_id": int(track_id), "segment_track_ids": segment_ids,
            "cut_frames": cuts,
            "evidence": [{key: value for key, value in row.items() if key not in {"index", "strength"}} for row in selected],
        })
    return TrackletSplitResult(
        output, output_positions, remap,
        {"method": "gta_link_inspired_temporal_kit_reid_change_point",
         "split_tracklets": len(split_rows), "new_tracklets": len(output), "splits": split_rows},
    )


def infer_team_assignments(
    tracklets: dict, ids: list[int], *, n_clusters: int = 3,
    min_train_frames: int = 60, min_train_samples: int = 4,
    max_distance: float = .35, min_margin: float = .08,
) -> dict[int, int]:
    """Infer only high-confidence kit groups; ambiguous tracks abstain.

    Centers are learned from longer, repeatedly sampled local tracks so that a
    one-frame foreground occluder cannot define a team.  Cluster numbers are
    deliberately opaque: they are used only as a pairwise hard-conflict gate.
    """
    train_ids = [
        track_id for track_id in ids
        if len(tracklets[track_id].get("frames", ())) >= min_train_frames
        and len(tracklets[track_id].get("team_feature_samples", ())) >= min_train_samples
        and np.any(np.asarray(tracklets[track_id].get("team_feature", []), np.float32))
    ]
    if n_clusters < 2 or len(train_ids) < n_clusters:
        return {}
    vectors = np.stack([
        np.asarray(tracklets[track_id]["team_feature"], np.float32) for track_id in train_ids
    ])
    cv2.setRNGSeed(7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    _, _, centers = cv2.kmeans(
        vectors, n_clusters, None, criteria, 20, cv2.KMEANS_PP_CENTERS,
    )
    assignments = {}
    for track_id in ids:
        samples = tracklets[track_id].get("team_feature_samples", ())
        feature = np.asarray(tracklets[track_id].get("team_feature", []), np.float32)
        if len(samples) < min_train_samples or not feature.size or not np.any(feature):
            continue
        distances = np.linalg.norm(centers - feature, axis=1)
        order = np.argsort(distances)
        margin = float(distances[order[1]] - distances[order[0]])
        if float(distances[order[0]]) <= max_distance and margin >= min_margin:
            assignments[track_id] = int(order[0])
    return assignments


def _kit_conflict(left: KitDescriptor, right: KitDescriptor, config: AssociationConfig) -> bool:
    if min(left.confidence, right.confidence) < config.kit_confidence_min:
        return False
    if left.family == "unknown" or right.family == "unknown":
        return False
    if left.family != right.family:
        return True
    return (
        left.family == "colour"
        and _circular_degrees(float(left.hue_degrees), float(right.hue_degrees))
        >= config.kit_hue_conflict_degrees
    )


def _same_kit(left: KitDescriptor, right: KitDescriptor, config: AssociationConfig) -> bool:
    if min(left.confidence, right.confidence) < config.kit_confidence_min:
        return False
    if left.family != right.family or left.family == "unknown":
        return False
    if left.family == "white":
        return True
    return _circular_degrees(float(left.hue_degrees), float(right.hue_degrees)) <= config.kit_hue_same_degrees


def _endpoint(points: list[tuple[int, float, float]], *, at_end: bool) -> np.ndarray | None:
    if not points:
        return None
    ordered = sorted(points)
    window = ordered[-5:] if at_end else ordered[:5]
    return np.median(np.asarray([[row[1], row[2]] for row in window], np.float64), axis=0)


def evaluate_pair(
    left_id: int, right_id: int, tracklets: dict, metric_positions: dict[int, list[tuple[int, float, float]]],
    processed_fps: float, config: AssociationConfig,
    team_assignments: dict[int, int] | None = None,
    prepared: dict | None = None,
) -> PairDecision:
    left, right = tracklets[left_id], tracklets[right_id]
    left_frames = prepared["frames"][left_id] if prepared else set(left["frames"])
    right_frames = prepared["frames"][right_id] if prepared else set(right["frames"])
    if left_frames & right_frames:
        return PairDecision(left_id, right_id, False, "temporal_overlap")

    kit_left = prepared["kits"][left_id] if prepared else describe_kit(left)
    kit_right = prepared["kits"][right_id] if prepared else describe_kit(right)
    if _kit_conflict(kit_left, kit_right, config):
        return PairDecision(left_id, right_id, False, "high_confidence_kit_conflict")
    stable_left = prepared["stable_colours"][left_id] if prepared else describe_stable_colour(left)
    stable_right = prepared["stable_colours"][right_id] if prepared else describe_stable_colour(right)
    if _kit_conflict(stable_left, stable_right, config):
        return PairDecision(left_id, right_id, False, "stable_colour_kit_conflict")
    if (team_assignments and left_id in team_assignments and right_id in team_assignments
            and team_assignments[left_id] != team_assignments[right_id]):
        return PairDecision(left_id, right_id, False, "automatic_team_cluster_conflict")

    if max(left_frames) < min(right_frames):
        earlier_id, later_id = left_id, right_id
        gap_frames = min(right_frames) - max(left_frames)
    else:
        earlier_id, later_id = right_id, left_id
        gap_frames = min(left_frames) - max(right_frames)
    gap_seconds = gap_frames / max(float(processed_fps), 1e-9)
    if gap_seconds > config.maximum_merge_gap_seconds:
        return PairDecision(
            left_id, right_id, False, "gap_beyond_maximum", gap_seconds=gap_seconds,
        )
    tier = "short" if gap_seconds <= config.short_gap_seconds else "medium" if gap_seconds <= config.long_gap_seconds else "long"

    distance = speed = None
    if tier == "short":
        end = _endpoint(metric_positions.get(earlier_id, []), at_end=True)
        start = _endpoint(metric_positions.get(later_id, []), at_end=False)
        if end is None or start is None:
            return PairDecision(left_id, right_id, False, "missing_metric_endpoint", gap_seconds=gap_seconds, tier=tier)
        distance = float(np.linalg.norm(start - end))
        speed = distance / max(gap_seconds, 1.0 / max(float(processed_fps), 1e-9))
        allowed = config.maximum_speed_mps * gap_seconds + config.metric_uncertainty_m
        if distance > allowed:
            return PairDecision(
                left_id, right_id, False, "short_gap_impossible_metric_speed",
                gap_seconds, distance, speed, tier=tier,
            )
    elif tier == "long" and not _same_kit(kit_left, kit_right, config):
        return PairDecision(
            left_id, right_id, False, "long_gap_without_confident_same_kit",
            gap_seconds=gap_seconds, tier=tier,
        )

    embedding_left, colour_left = prepared["features"][left_id] if prepared else _normalised_feature(left)
    embedding_right, colour_right = prepared["features"][right_id] if prepared else _normalised_feature(right)
    if (not embedding_left.size or not embedding_right.size or not colour_left.size or not colour_right.size
            or embedding_left.shape != embedding_right.shape
            or np.linalg.norm(embedding_left) < 1e-9 or np.linalg.norm(embedding_right) < 1e-9):
        return PairDecision(left_id, right_id, False, "missing_appearance", gap_seconds=gap_seconds, tier=tier)
    embedding_similarity = float(np.dot(embedding_left, embedding_right))
    colour_similarity = float(cv2.compareHist(
        colour_left.reshape(-1, 1), colour_right.reshape(-1, 1), cv2.HISTCMP_CORREL
    ))
    if colour_similarity < config.colour_min:
        return PairDecision(
            left_id, right_id, False, "colour_similarity_below_minimum", gap_seconds,
            distance, speed, embedding_similarity, colour_similarity, tier=tier,
        )

    reid_min = {"short": config.short_reid_min, "medium": config.medium_reid_min, "long": config.long_reid_min}[tier]
    if embedding_similarity < reid_min:
        return PairDecision(
            left_id, right_id, False, "reid_below_time_tier_threshold", gap_seconds,
            distance, speed, embedding_similarity, colour_similarity, tier=tier,
        )
    score = config.appearance_weight * embedding_similarity + config.colour_weight * colour_similarity
    score_min = {"short": config.short_score_min, "medium": config.medium_score_min, "long": config.long_score_min}[tier]
    if score < score_min:
        return PairDecision(
            left_id, right_id, False, "combined_score_below_time_tier_threshold", gap_seconds,
            distance, speed, embedding_similarity, colour_similarity, score, tier,
        )
    return PairDecision(
        left_id, right_id, True, "passed_hard_gates_and_reid", gap_seconds,
        distance, speed, embedding_similarity, colour_similarity, score, tier,
    )


def associate_tracklets(
    tracklets: dict, total_frames: int, processed_fps: float,
    metric_positions: dict[int, list[tuple[int, float, float]]], config: AssociationConfig,
) -> AssociationResult:
    all_ids = sorted(
        track_id for track_id, tracklet in tracklets.items()
        if tracklet.get("frames")
    )
    ids = [
        track_id for track_id in all_ids
        if len(tracklets[track_id]["frames"]) >= config.min_track_frames
        and not tracklets[track_id].get("association_quarantine", False)
    ]
    team_assignments = infer_team_assignments(
        tracklets, ids, n_clusters=config.team_clusters,
        min_train_frames=config.team_min_train_frames,
        min_train_samples=config.team_min_train_samples,
        max_distance=config.team_max_distance, min_margin=config.team_min_margin,
    )
    prepared = {
        "frames": {track_id: set(tracklets[track_id]["frames"]) for track_id in ids},
        "kits": {track_id: describe_kit(tracklets[track_id]) for track_id in ids},
        "stable_colours": {track_id: describe_stable_colour(tracklets[track_id]) for track_id in ids},
        "features": {track_id: _normalised_feature(tracklets[track_id]) for track_id in ids},
    }
    decisions: list[PairDecision] = []
    rejected = Counter()
    accepted_raw: list[PairDecision] = []
    for left_index, left_id in enumerate(ids):
        for right_id in ids[left_index + 1:]:
            decision = evaluate_pair(
                left_id, right_id, tracklets, metric_positions, processed_fps, config,
                team_assignments, prepared,
            )
            decisions.append(decision)
            if decision.accepted:
                accepted_raw.append(decision)
            else:
                rejected[decision.reason] += 1

    neighbours: dict[int, list[int]] = defaultdict(list)
    by_identity: dict[int, list[PairDecision]] = defaultdict(list)
    for decision in accepted_raw:
        by_identity[decision.left].append(decision)
        by_identity[decision.right].append(decision)
    for track_id, rows in by_identity.items():
        rows.sort(key=lambda row: float(row.score), reverse=True)
        neighbours[track_id] = [row.right if row.left == track_id else row.left for row in rows[:config.reciprocal_top_k]]

    reciprocal = [
        row for row in accepted_raw
        if row.right in neighbours[row.left] and row.left in neighbours[row.right]
    ]
    rejected["not_reciprocal_top_k"] += len(accepted_raw) - len(reciprocal)
    reciprocal.sort(key=lambda row: float(row.score), reverse=True)
    # Reciprocal top-k is required only for the edge that triggers a merge.
    # Complete-link then checks every cross-cluster pair against the full set
    # that passed all hard gates and ReID.  Using only reciprocal edges for the
    # complete-link matrix would accidentally cap a cluster near top_k + 1.
    pair_evidence = {(min(row.left, row.right), max(row.left, row.right)): row for row in accepted_raw}

    parent = {track_id: track_id for track_id in ids}
    members = {track_id: {track_id} for track_id in ids}
    frame_sets = {track_id: set(tracklets[track_id]["frames"]) for track_id in ids}

    def find(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    merges = []
    complete_link_rejections = 0
    for trigger in reciprocal:
        left_root, right_root = find(trigger.left), find(trigger.right)
        if left_root == right_root:
            continue
        if frame_sets[left_root] & frame_sets[right_root]:
            rejected["cluster_temporal_overlap"] += 1
            continue
        cross = []
        for left_member in members[left_root]:
            for right_member in members[right_root]:
                evidence = pair_evidence.get((min(left_member, right_member), max(left_member, right_member)))
                if evidence is None:
                    cross = []
                    break
                cross.append(evidence)
            if not cross:
                break
        if not cross:
            complete_link_rejections += 1
            continue
        if len(members[left_root]) < len(members[right_root]):
            left_root, right_root = right_root, left_root
        before_left, before_right = sorted(members[left_root]), sorted(members[right_root])
        parent[right_root] = left_root
        members[left_root] |= members[right_root]
        frame_sets[left_root] |= frame_sets[right_root]
        merges.append({
            "trigger_pair": [trigger.left, trigger.right], "trigger_score": trigger.score,
            "complete_link_min_score": min(float(row.score) for row in cross),
            "left_members": before_left, "right_members": before_right,
            "result_members": sorted(members[left_root]),
        })

    cluster_frames: dict[int, set[int]] = defaultdict(set)
    cluster_members: dict[int, list[int]] = defaultdict(list)
    for track_id in ids:
        root = find(track_id)
        cluster_frames[root] |= set(tracklets[track_id]["frames"])
        cluster_members[root].append(track_id)
    # A caller may still raise `min_track_frames` for an experiment. It must
    # never silently remove a field-gated tracklet from the delivered MOT:
    # ineligible tracklets remain independent singleton identities.
    for track_id in all_ids:
        if track_id not in parent:
            cluster_frames[track_id] |= set(tracklets[track_id]["frames"])
            cluster_members[track_id].append(track_id)
    below_presence = sum(
        len(frames) / max(int(total_frames), 1) < config.min_presence_ratio
        for frames in cluster_frames.values()
    )
    # Presence is diagnostic/ranking metadata, never a deletion rule. Every
    # field-valid tracklet must receive a global ID, merged or singleton.
    kept = list(cluster_frames)
    kept.sort(key=lambda root: (len(cluster_frames[root]), -min(cluster_members[root])), reverse=True)
    root_to_global = {root: global_id for global_id, root in enumerate(kept)}
    mapping = {}
    for track_id in all_ids:
        root = find(track_id) if track_id in parent else track_id
        if root in root_to_global:
            mapping[track_id] = root_to_global[root]
    audit = {
        "policy": "hard_time_kit_metric_then_tiered_reid_reciprocal_complete_link",
        "parameters": asdict(config),
        "counts": {
            "input_tracklets": len(all_ids), "eligible_tracklets": len(ids),
            "evaluated_pairs": len(decisions), "passed_hard_gates_and_reid": len(accepted_raw),
            "reciprocal_candidate_pairs": len(reciprocal), "accepted_merges": len(merges),
            "complete_link_rejections": complete_link_rejections,
            "clusters_before_presence_filter": len(cluster_frames), "kept_global_ids": len(kept),
            "clusters_below_legacy_presence_threshold": below_presence,
            "rejected_by_reason": dict(sorted(rejected.items())),
            "high_confidence_team_assignments": len(team_assignments),
        },
        "team_assignments": {str(key): value for key, value in sorted(team_assignments.items())},
        "merges": merges,
    }
    return AssociationResult(mapping, audit)
