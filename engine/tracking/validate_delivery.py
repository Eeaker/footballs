from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2


V3_REFERENCE = {
    "total_processed_frames": 33829,
    "mot_rows": 353317,
    "mean_boxes_per_frame": 10.4442,
    "global_ids": 43,
    "event_candidates": 54,
    "id_focus_clips": 48,
    "duplicate_same_frame_id_extra_rows": 0,
}


def mot_metrics(path: Path, total_frames: int) -> dict:
    counts = Counter()
    seen = set()
    duplicate_rows = 0
    ids = set()
    rows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        frame, gid = int(float(parts[0])), int(float(parts[1]))
        rows += 1
        counts[frame] += 1
        ids.add(gid)
        key = (frame, gid)
        duplicate_rows += int(key in seen)
        seen.add(key)
    ordered = sorted(counts.values())
    return {
        "mot_rows": rows,
        "frames_with_boxes": len(counts),
        "empty_frames": max(0, total_frames - len(counts)),
        "mean_boxes_per_frame": round(rows / max(total_frames, 1), 4),
        "median_boxes_per_frame": ordered[len(ordered) // 2] if ordered else 0,
        "max_boxes_per_frame": max(ordered, default=0),
        "global_ids": len(ids),
        "duplicate_same_frame_id_extra_rows": duplicate_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="验收tracking V3交付结果")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracking-dir", type=Path, required=True)
    parser.add_argument("--id-focus-dir", type=Path, required=True)
    parser.add_argument("--mosaics-dir", type=Path, required=True)
    parser.add_argument("--highlights-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["v3_exact", "adaptive"], default="v3_exact")
    parser.add_argument("--expected-highlights", type=int, default=5)
    args = parser.parse_args()

    metadata = json.loads((args.tracking_dir / "tracking_run_metadata.json").read_text(encoding="utf-8"))
    total_frames = int(metadata["total_processed_frames"])
    metrics = mot_metrics(args.tracking_dir / "tracking_mot.txt", total_frames)
    events = json.loads((args.tracking_dir / "events.json").read_text(encoding="utf-8"))
    focus = json.loads((args.id_focus_dir / "id_focus_clips.json").read_text(encoding="utf-8"))
    mosaics = json.loads((args.mosaics_dir / "identity_mosaics_manifest.json").read_text(encoding="utf-8"))
    highlights = json.loads((args.highlights_dir / "five_event_candidate_manifest.json").read_text(encoding="utf-8"))

    checks = {
        "tracking_has_player_boxes": metrics["frames_with_boxes"] > 0,
        "no_same_frame_duplicate_global_id": metrics["duplicate_same_frame_id_extra_rows"] == 0,
        "candidate_ids_present": metrics["global_ids"] > 0,
        "identity_mosaic_count_matches_ids": mosaics["global_id_count"] == metrics["global_ids"],
        "id_focus_manifest_present": isinstance(focus, list),
        "expected_highlight_candidates_present": highlights["event_count"] == min(args.expected_highlights, len(events)),
        "no_direct_multimodal_scoring": metadata.get("multimodal_direct_scoring") == "frozen_not_in_default_pipeline",
    }
    if args.profile == "v3_exact":
        checks.update({
            "video_frame_count_matches_v3": total_frames == V3_REFERENCE["total_processed_frames"],
            "all_frames_have_player_boxes": metrics["frames_with_boxes"] == total_frames,
            "mean_boxes_not_below_v3_tolerance": metrics["mean_boxes_per_frame"] >= 9.9,
            "candidate_id_count_in_safe_band": 35 <= metrics["global_ids"] <= 50,
            "event_count_matches_v3": len(events) == V3_REFERENCE["event_candidates"],
            "id_focus_clips_match_v3": len(focus) == V3_REFERENCE["id_focus_clips"],
        })
    comparison = {}
    actual = {**metrics, "total_processed_frames": total_frames, "event_candidates": len(events),
              "id_focus_clips": len(focus)}
    for key, reference in V3_REFERENCE.items():
        value = actual[key]
        comparison[key] = {
            "v3_reference": reference,
            "actual": value,
            "delta": round(value - reference, 4),
        }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "profile": args.profile,
        "checks": checks,
        "metrics": metrics,
        "artifacts": {
            "event_candidates": len(events),
            "id_focus_clips": len(focus),
            "identity_mosaics": mosaics["global_id_count"],
            "highlight_candidates": highlights["event_count"],
        },
        "v3_comparison": comparison,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
