from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRACKING_ROOT = Path(__file__).resolve().parents[3] / "tracking"
sys.path.insert(0, str(ROOT))
if str(TRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKING_ROOT))

from tracking_lib.team_features import jersey_feature
from mode_split.transitions import ModeObservation, detect_persistent_transitions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mot(
    path: Path, selected_gids: set[int] | None = None,
    frame_start: int = 1, frame_end: int | None = None,
) -> tuple[dict[int, list[tuple]], set[int]]:
    by_frame: dict[int, list[tuple]] = defaultdict(list)
    gids: set[int] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            values = line.rstrip("\n").split(",")
            if len(values) < 7:
                raise ValueError(f"MOT line {line_number} has fewer than seven columns")
            frame, gid = int(float(values[0])), int(float(values[1]))
            if frame < frame_start or (frame_end is not None and frame > frame_end):
                continue
            if selected_gids is not None and gid not in selected_gids:
                continue
            x, y, w, h = map(float, values[2:6])
            by_frame[frame].append((gid, x, y, w, h))
            gids.add(gid)
    return dict(by_frame), gids


def box_feature(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    x, y, w, h = box
    height, width = frame.shape[:2]
    x1 = max(0, min(width, int(round(x + .24 * w))))
    x2 = max(0, min(width, int(round(x + .76 * w))))
    y1 = max(0, min(height, int(round(y + .10 * h))))
    y2 = max(0, min(height, int(round(y + .58 * h))))
    torso = frame[y1:y2, x1:x2]
    if torso.size == 0 or min(torso.shape[:2]) < 3:
        return None
    scale = min(1.0, 32.0 / max(torso.shape[:2]))
    if scale < 1.0:
        torso = cv2.resize(torso, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return jersey_feature(torso)


def learn_modes(
    video: Path, by_frame: dict[int, list[tuple]], *, clusters: int,
    sample_stride_frames: int, frame_start: int = 1, frame_end: int | None = None,
) -> tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_end or 10**18)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start - 1)
    samples: list[np.ndarray] = []
    for frame_id in range(frame_start, total + 1):
        ok, image = cap.read()
        if not ok:
            break
        if frame_id % sample_stride_frames:
            continue
        for gid, x, y, w, h in by_frame.get(frame_id, []):
            feature = box_feature(image, (x, y, w, h))
            if feature is not None:
                samples.append(feature)
    cap.release()
    if len(samples) < clusters * 10:
        raise RuntimeError(f"too few mode-learning samples: {len(samples)}")
    matrix = np.asarray(samples, np.float32)
    cv2.setRNGSeed(20260824)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-5)
    _, _, centers = cv2.kmeans(matrix, clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-9)
    return centers.astype(np.float32), len(samples)


def classify(feature: np.ndarray | None, centers: np.ndarray) -> tuple[int | None, float, float]:
    if feature is None:
        return None, 0.0, float("inf")
    distances = np.linalg.norm(centers - feature[None, :], axis=1)
    order = np.argsort(distances)
    first, second = float(distances[order[0]]), float(distances[order[1]])
    confidence = max(0.0, (second - first) / max(second, 1e-9))
    return int(order[0]), confidence, first


def observe_all_boxes(
    video: Path, by_frame: dict[int, list[tuple]], centers: np.ndarray,
    output_csv: Path, frame_start: int = 1, frame_end: int | None = None,
) -> tuple[dict[int, list[ModeObservation]], int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_end or 10**18)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start - 1)
    grouped: dict[int, list[ModeObservation]] = defaultdict(list)
    written = 0
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "global_id", "self_mode", "mode_confidence", "center_distance"])
        for frame_id in range(frame_start, total + 1):
            ok, image = cap.read()
            if not ok:
                break
            for gid, x, y, w, h in by_frame.get(frame_id, []):
                mode, confidence, distance = classify(box_feature(image, (x, y, w, h)), centers)
                grouped[gid].append(ModeObservation(frame_id, mode, confidence))
                writer.writerow([
                    frame_id, gid, "" if mode is None else mode,
                    f"{confidence:.8f}", "" if not np.isfinite(distance) else f"{distance:.8f}",
                ])
                written += 1
    cap.release()
    return dict(grouped), written


def mode_summary(rows: Iterable[ModeObservation], minimum_confidence: float) -> dict[str, int]:
    counts = Counter(
        f"mode_{row.mode}" if row.mode is not None and row.confidence >= minimum_confidence else "unknown"
        for row in rows
    )
    return dict(sorted(counts.items()))


def write_split_mot(
    source: Path, destination: Path, cuts_by_gid: dict[int, list[int]],
) -> list[dict]:
    all_gids: set[int] = set()
    with source.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            values = line.split(",")
            if len(values) >= 2:
                all_gids.add(int(float(values[1])))
    next_gid = max(all_gids, default=-1) + 1
    mappings: dict[tuple[int, int], int] = {}
    records: dict[tuple[int, int], dict] = {}
    for gid in sorted(all_gids):
        cut_frames = sorted(set(cuts_by_gid.get(gid, [])))
        for segment_index in range(len(cut_frames) + 1):
            segment_gid = gid if segment_index == 0 else next_gid
            if segment_index:
                next_gid += 1
            mappings[(gid, segment_index)] = segment_gid
            records[(gid, segment_index)] = {
                "original_global_id": gid, "segment_index": segment_index,
                "segment_global_id": segment_gid,
                "start_frame": None, "end_frame": None,
            }
    with source.open("r", encoding="utf-8-sig") as source_handle, \
            destination.open("w", encoding="utf-8", newline="") as output_handle:
        for line in source_handle:
            values = line.rstrip("\n").split(",")
            frame, gid = int(float(values[0])), int(float(values[1]))
            segment_index = sum(frame >= cut for cut in sorted(set(cuts_by_gid.get(gid, []))))
            segment_gid = mappings[(gid, segment_index)]
            values[1] = str(segment_gid)
            output_handle.write(",".join(values) + "\n")
            record = records[(gid, segment_index)]
            record["start_frame"] = frame if record["start_frame"] is None else min(record["start_frame"], frame)
            record["end_frame"] = frame if record["end_frame"] is None else max(record["end_frame"], frame)
    return [record for record in records.values() if record["start_frame"] is not None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--sample-stride-frames", type=int, default=30)
    parser.add_argument("--mode-confidence", type=float, default=.12)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--minimum-evidence", type=int, default=5)
    parser.add_argument("--purity", type=float, default=.75)
    parser.add_argument("--reversal-horizon-frames", type=int, default=30)
    parser.add_argument("--only-gids", type=int, nargs="*")
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video, mot, output = args.video.resolve(), args.mot.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output must not exist: {output}")
    output.mkdir(parents=True)
    selected = set(args.only_gids) if args.only_gids else None
    if args.frame_start < 1 or (args.frame_end is not None and args.frame_end < args.frame_start):
        raise ValueError("invalid frame range")
    by_frame, selected_gids = read_mot(
        mot, selected, frame_start=args.frame_start, frame_end=args.frame_end,
    )
    # Learn run-local modes from all detections in the same time range.  The
    # optional gid filter limits only the audited targets; it must not turn
    # one player's lighting/pose variation into artificial "team" modes.
    learning_by_frame = by_frame
    if selected is not None:
        learning_by_frame, _ = read_mot(
            mot, None, frame_start=args.frame_start, frame_end=args.frame_end,
        )
    centers, learning_samples = learn_modes(
        video, learning_by_frame, clusters=args.clusters,
        sample_stride_frames=args.sample_stride_frames,
        frame_start=args.frame_start, frame_end=args.frame_end,
    )
    np.save(output / "self_mode_centers.npy", centers)
    grouped, observation_count = observe_all_boxes(
        video, by_frame, centers, output / "mode_observations.csv",
        frame_start=args.frame_start, frame_end=args.frame_end,
    )
    transitions = {}
    for gid, rows in grouped.items():
        transitions[gid] = detect_persistent_transitions(
            rows, window=args.window, minimum_evidence=args.minimum_evidence,
            purity=args.purity, minimum_confidence=args.mode_confidence,
            reversal_horizon_frames=args.reversal_horizon_frames,
        )
    cuts_by_gid = {gid: [row.frame for row in rows] for gid, rows in transitions.items() if rows}
    mapping = []
    if selected is None:
        mapping = write_split_mot(mot, output / "tracking_mot_self_mode_split.txt", cuts_by_gid)
        with (output / "segment_map.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mapping[0]))
            writer.writeheader()
            writer.writerows(mapping)
    transition_rows = []
    for gid, rows in sorted(transitions.items()):
        for row in rows:
            transition_rows.append({
                "global_id": gid, "cut_frame": row.frame,
                "before_mode": row.before_mode, "after_mode": row.after_mode,
                "before_purity": round(row.before_purity, 6),
                "after_purity": round(row.after_purity, 6),
                "before_evidence": row.before_evidence,
                "after_evidence": row.after_evidence,
            })
    with (output / "transitions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["global_id", "cut_frame", "before_mode", "after_mode", "before_purity",
                  "after_purity", "before_evidence", "after_evidence"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(transition_rows)
    gid11_cuts = cuts_by_gid.get(11, [])
    report = {
        "schema_version": 1,
        "status": "experimental_not_for_delivery",
        "policy": "per-detection self-derived unsupervised kit mode; no semantic colour input",
        "sources": {
            "video": str(video), "mot": str(mot),
            "mot_sha256": sha256(mot),
        },
        "parameters": {
            "clusters": args.clusters, "sample_stride_frames": args.sample_stride_frames,
            "mode_confidence": args.mode_confidence, "window": args.window,
            "minimum_evidence": args.minimum_evidence, "purity": args.purity,
            "reversal_horizon_frames": args.reversal_horizon_frames,
            "frame_start": args.frame_start, "frame_end": args.frame_end,
            "max_ids_merge_policy": "unchanged_upstream_result",
        },
        "counts": {
            "input_global_ids": len(selected_gids), "learning_samples": learning_samples,
            "box_observations": observation_count, "detected_transitions": len(transition_rows),
            "ids_with_transitions": len(cuts_by_gid),
            "output_segments": len(mapping) if mapping else None,
        },
        "mode_counts_by_global_id": {
            str(gid): mode_summary(rows, args.mode_confidence) for gid, rows in sorted(grouped.items())
        },
        "known_regression": {
            "global_id": 11, "expected_switch_window": [28168, 28184],
            "detected_cuts": gid11_cuts,
            "passed": any(28168 <= frame <= 28184 for frame in gid11_cuts),
        },
        "limitations": [
            "This experiment detects cross-mode contamination; it does not prove same-team identity.",
            "Mode indices are run-local and have no manually supplied colour meaning.",
            "The upstream greedy max_ids=10 merge policy is deliberately unchanged for this A/B.",
        ],
    }
    (output / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report["counts"] | {"known_regression": report["known_regression"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
