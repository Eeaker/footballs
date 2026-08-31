"""Adapter from MOT tracklets to the migrated MIT-licensed JerseyOCR backend.

The recognizer and its crop-level deduplication/voting logic are imported from
``third_party/football-player-identification`` at its pinned upstream commit.
This module only adapts our MOT rows, applies the conservative identity gate,
and emits the verifier format consumed by the player-card exporter.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


UPSTREAM_COMMIT = "b1bd36428ba55ed970ebda01d17559b9cd044bb6"
UPSTREAM_REPOSITORY = "https://github.com/Cappetti99/football-player-identification.git"


def _upstream_root() -> Path:
    return Path(__file__).resolve().parent.parent / "third_party" / "football-player-identification"


def load_upstream() -> tuple[type, Any]:
    root = _upstream_root()
    if not (root / "ft" / "features" / "jersey_ocr.py").is_file():
        raise FileNotFoundError(f"migrated JerseyOCR source missing: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from ft.export.artifacts import crop_quality
    from ft.features.jersey_ocr import JerseyOCR

    return JerseyOCR, crop_quality


def read_mot_deduplicated(path: str | Path) -> tuple[list[dict], int]:
    selected: dict[tuple[int, int], dict] = {}
    duplicates = 0
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            values = line.split(",")
            if len(values) < 7:
                raise ValueError(f"MOT line {line_number} has fewer than seven columns")
            row = {
                "frame": int(float(values[0])) - 1, "global_id": int(float(values[1])),
                "x": float(values[2]), "y": float(values[3]),
                "w": float(values[4]), "h": float(values[5]),
                "confidence": float(values[6]),
            }
            key = (row["frame"], row["global_id"])
            previous = selected.get(key)
            if previous is not None:
                duplicates += 1
            if previous is None or row["confidence"] > previous["confidence"]:
                selected[key] = row
    if not selected:
        raise ValueError(f"MOT is empty: {path}")
    return sorted(selected.values(), key=lambda row: (row["frame"], row["global_id"])), duplicates


def load_team_hints(path: str | Path) -> dict[int, str]:
    path = Path(path)
    result: dict[int, str] = {}
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                team = str(row.get("team") or row.get("team_id") or "").strip().lower()
                if team:
                    result[int(row["global_id"])] = team
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        for bucket in ("eligible_confirmed", "excluded_conflict", "excluded_mismatch",
                       "excluded_unreadable", "excluded_nonplayer"):
            for row in data.get(bucket, []):
                team = str(row.get("team") or "").strip().lower()
                if team:
                    result[int(row["global_id"])] = team
    if not result:
        raise ValueError(f"team hints contain no global_id mapping: {path}")
    return result


def _candidate_pool(rows: list[dict], maximum_per_id: int) -> dict[int, list[dict]]:
    by_id: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["w"] >= 18 and row["h"] >= 36:
            by_id[row["global_id"]].append(row)
    selected: dict[int, list[dict]] = {}
    for gid, track in by_id.items():
        track.sort(key=lambda row: row["frame"])
        bucket_count = min(maximum_per_id, len(track))
        candidates = []
        for index in range(bucket_count):
            start = int(round(index * len(track) / bucket_count))
            end = max(start + 1, int(round((index + 1) * len(track) / bucket_count)))
            bucket = track[start:end]
            candidates.append(max(bucket, key=lambda row: (row["w"] * row["h"], row["confidence"])))
        selected[gid] = candidates
    return selected


def extract_tracklet_crops(
    *, video: str | Path, mot_rows: list[dict], output: str | Path, maximum_per_id: int = 36,
) -> tuple[list[dict], dict]:
    _, upstream_crop_quality = load_upstream()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    selected = _candidate_pool(mot_rows, maximum_per_id)
    requests: dict[int, list[dict]] = defaultdict(list)
    for rows in selected.values():
        for row in rows:
            requests[row["frame"]].append(row)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    adapted = []
    failed_frames = []
    for frame_index in sorted(requests):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            failed_frames.append(frame_index)
            continue
        height, width = frame.shape[:2]
        for row in requests[frame_index]:
            padding_x, padding_y = row["w"] * 0.04, row["h"] * 0.02
            bbox = [
                max(0.0, row["x"] - padding_x), max(0.0, row["y"] - padding_y),
                min(float(width), row["x"] + row["w"] + padding_x),
                min(float(height), row["y"] + row["h"] + padding_y),
            ]
            x1, y1, x2, y2 = [int(round(value)) for value in bbox]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            gid_dir = output / f"gid_{row['global_id']:03d}"
            gid_dir.mkdir(exist_ok=True)
            crop_path = gid_dir / f"frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                continue
            adapted.append({
                "frame": frame_index, "track_id": row["global_id"],
                "display_track_id": row["global_id"], "role_detection": "player",
                "crop_path": str(crop_path.resolve()), "bbox": bbox,
                "crop_quality": float(upstream_crop_quality(bbox, frame)),
                "tracking_confidence": row["confidence"],
            })
    cap.release()
    return adapted, {
        "candidate_global_ids": len(selected), "candidate_rows": sum(map(len, selected.values())),
        "written_crops": len(adapted), "failed_video_frames": failed_frames,
    }


def load_existing_tracklet_crops(path: str | Path) -> tuple[list[dict], dict]:
    """Resume OCR from an immutable candidate directory without decoding the video again."""
    path = Path(path).resolve()
    rows = []
    for gid_dir in sorted(path.glob("gid_*")):
        try:
            gid = int(gid_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        for crop_path in sorted(gid_dir.glob("frame_*.jpg")):
            try:
                frame = int(crop_path.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            image = cv2.imread(str(crop_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            rows.append({
                "frame": frame, "track_id": gid, "display_track_id": gid,
                "role_detection": "player", "crop_path": str(crop_path),
                "bbox": [0.0, 0.0, float(width), float(height)], "crop_quality": 1.0,
            })
    if not rows:
        raise ValueError(f"candidate directory contains no readable crops: {path}")
    return rows, {
        "reused_candidate_directory": str(path), "written_crops": len(rows),
        "candidate_global_ids": len({row["display_track_id"] for row in rows}),
    }


def conservative_status(voted: dict | None, diagnostic: dict | None) -> tuple[str, str]:
    """Convert upstream evidence into player-card verifier status without guessing."""
    if not voted:
        return "unreadable", "upstream_abstained"
    candidates = voted.get("candidates") or []
    runner = candidates[1] if len(candidates) > 1 else None
    if runner and int(runner.get("votes", 0)) >= 2 and float(voted.get("winner_margin", 0.0)) < 0.15:
        return "conflict", "multiple_numbers_with_independent_support"
    hard_gate = (
        int(voted.get("votes", 0)) >= 5
        and float(voted.get("confidence", 0.0)) >= 0.20
        and float(voted.get("head_confidence", 0.0)) >= 0.55
        and float(voted.get("winner_margin", 0.0)) >= 0.10
        and bool(voted.get("full_body_sufficient", False))
    )
    if hard_gate:
        return "confirmed", "mit_upstream_vote_plus_conservative_identity_gate"
    reason = (diagnostic or {}).get("decision", {}).get("status") or "identity_gate_not_met"
    return "unreadable", str(reason)


def _simultaneous_identity_conflicts(results: list[dict], mot_rows: list[dict]) -> set[int]:
    frames_by_id: dict[int, set[int]] = defaultdict(set)
    for row in mot_rows:
        frames_by_id[row["global_id"]].add(row["frame"])
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in results:
        if row["status"] == "confirmed":
            groups[(row["team"], int(row["predicted_number"]))].append(row["global_id"])
    conflicts = set()
    for gids in groups.values():
        for index, left in enumerate(gids):
            for right in gids[index + 1:]:
                if frames_by_id[left] & frames_by_id[right]:
                    conflicts.update((left, right))
    return conflicts


def adapt_number_results_csv(
    *, numbers: str | Path, mot: str | Path, output: str | Path,
    team_hints: str | Path | None = None,
) -> Path:
    """Adapt the v1.0 documented jersey_number_results.csv into verifier JSON."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output must not exist: {output}")
    teams = load_team_hints(team_hints) if team_hints else {}
    rows = []
    with Path(numbers).open("r", encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            gid = int(source["global_id"])
            status = str(source.get("status") or "unreadable").strip().lower()
            number_value = str(source.get("predicted_number") or source.get("final_number") or "").strip()
            team = str(source.get("team") or teams.get(gid) or "unassigned").strip().lower()
            rows.append({
                "global_id": gid, "team": team,
                "predicted_number": int(float(number_value)) if number_value else None,
                "confidence": float(source.get("confidence") or 0.0),
                "status": "confirmed" if status in {"confirmed", "eligible_confirmed"} and number_value else
                          "conflict" if status in {"conflict", "excluded_conflict"} else "unreadable",
                "source_status": status,
            })
    if not rows:
        raise ValueError(f"number results CSV is empty: {numbers}")
    missing_teams = [row["global_id"] for row in rows if row["team"] == "unassigned"]
    if missing_teams:
        raise ValueError(f"team mapping missing for global_ids: {missing_teams}")
    mot_rows, _ = read_mot_deduplicated(mot)
    simultaneous = _simultaneous_identity_conflicts(rows, mot_rows)
    for row in rows:
        if row["global_id"] in simultaneous:
            row.update(status="conflict", predicted_number=None,
                       source_status="same_team_number_visible_simultaneously")
    eligibility = {
        "schema_version": "clip-eligibility-v1",
        "eligible_confirmed": [{
            "global_id": row["global_id"], "team": row["team"],
            "final_number": row["predicted_number"], "confidence": row["confidence"],
            "status": "eligible_confirmed", "source_status": row["source_status"],
        } for row in rows if row["status"] == "confirmed"],
        "excluded_conflict": [row for row in rows if row["status"] == "conflict"],
        "excluded_mismatch": [],
        "excluded_unreadable": [row for row in rows if row["status"] == "unreadable"],
        "provenance": {"source": str(Path(numbers).resolve()),
                       "policy": "v1.0 CSV adapter + simultaneous same-team/number conflict gate"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(eligibility, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def run_jersey_number_recognition(
    *, video: str | Path, mot: str | Path, team_hints: str | Path, output: str | Path,
    gpu: bool = True, maximum_candidates_per_id: int = 36,
    reuse_candidates: str | Path | None = None,
) -> dict:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output must not exist: {output}")
    output.mkdir(parents=True)
    mot_rows, duplicates = read_mot_deduplicated(mot)
    teams = load_team_hints(team_hints)
    if reuse_candidates:
        crop_rows, extraction = load_existing_tracklet_crops(reuse_candidates)
    else:
        crop_rows, extraction = extract_tracklet_crops(
            video=video, mot_rows=mot_rows, output=output / "candidates",
            maximum_per_id=maximum_candidates_per_id,
        )
    JerseyOCR, _ = load_upstream()
    recognizer = JerseyOCR(
        backend="easyocr", min_confidence=0.25, max_crops_per_tracklet=12,
        temporal_passes=1, augment=True, min_crop_quality=0.08, min_votes=2,
        min_raw_confidence=0.05, min_winner_margin=0.15, easyocr_gpu=gpu,
        aggregate_by_crop=True, max_candidates_per_crop=3, min_crop_candidate_ratio=0.35,
        cache_enabled=True, cache_dir=output / "cache", number_roi_enabled=False,
        demote_direct_only_single_digits=True, prefer_two_digit_candidates=True,
        strict_numeric_only=True, progress_every=5,
    )
    assignments, diagnostics = recognizer.recognize(crop_rows)
    diagnostics_by_id = diagnostics.get("tracklets", {})
    results = []
    all_gids = sorted({row["global_id"] for row in mot_rows})
    for gid in all_gids:
        voted = assignments.get(gid)
        diagnostic = diagnostics_by_id.get(str(gid)) or diagnostics_by_id.get(gid) or {}
        status, reason = conservative_status(voted, diagnostic)
        results.append({
            "global_id": gid, "team": teams.get(gid, "unassigned"),
            "predicted_number": int(voted["jersey_number"]) if status == "confirmed" else None,
            "confidence": round(float(voted.get("confidence", 0.0)), 8) if voted else 0.0,
            "status": status, "support_count": int(voted.get("votes", 0)) if voted else 0,
            "winner_margin": round(float(voted.get("winner_margin", 0.0)), 8) if voted else 0.0,
            "decision_rule": reason, "method": "MIT_upstream_JerseyOCR_EasyOCR_multiframe",
        })
    simultaneous = _simultaneous_identity_conflicts(results, mot_rows)
    for row in results:
        if row["global_id"] in simultaneous:
            row.update(status="conflict", predicted_number=None,
                       decision_rule="same_team_number_visible_simultaneously")

    with (output / "jersey_number_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    eligibility = {
        "schema_version": "clip-eligibility-v1",
        "eligible_confirmed": [{
            "global_id": row["global_id"], "team": row["team"],
            "final_number": row["predicted_number"], "confidence": row["confidence"],
            "status": "eligible_confirmed", "source_status": "confirmed",
        } for row in results if row["status"] == "confirmed" and row["team"] != "unassigned"],
        "excluded_conflict": [row for row in results if row["status"] == "conflict"],
        "excluded_mismatch": [],
        "excluded_unreadable": [row for row in results if row["status"] == "unreadable"],
        "provenance": {
            "upstream_repository": UPSTREAM_REPOSITORY, "upstream_commit": UPSTREAM_COMMIT,
            "upstream_license": "MIT", "backend": "EasyOCR 1.7.2",
            "policy": "crop-level deduplication + multiframe voting + conservative identity gate",
        },
    }
    (output / "clip_eligibility.json").write_text(
        json.dumps(eligibility, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "ocr_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, default=lambda value: value.item()
                   if isinstance(value, np.generic) else str(value)), encoding="utf-8",
    )
    manifest = {
        "video": str(Path(video).resolve()), "mot": str(Path(mot).resolve()),
        "team_hints": str(Path(team_hints).resolve()), "mot_duplicate_rows_removed": duplicates,
        "extraction": extraction, "global_ids": len(results),
        "confirmed": sum(row["status"] == "confirmed" for row in results),
        "conflict": sum(row["status"] == "conflict" for row in results),
        "unreadable": sum(row["status"] == "unreadable" for row in results),
        "outputs": {"results": "jersey_number_results.csv", "eligibility": "clip_eligibility.json",
                    "diagnostics": "ocr_diagnostics.json"},
    }
    (output / "number_recognition_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest
