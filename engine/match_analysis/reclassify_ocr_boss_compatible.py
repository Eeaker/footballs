from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按老板冻结交付的三态思路重判现有同场 OCR 证据，不重新跑模型",
    )
    parser.add_argument("--ocr-dir", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manual-audit", type=Path,
        help="可选人工看片结论 JSON；只允许显式保留、降级或标记 mismatch",
    )
    parser.add_argument(
        "--team-label", action="append", default=[], metavar="SOURCE=SEMANTIC",
        help="可重复，例如 team_1=blue、team_2=yellow、team_0=nonplayer",
    )
    return parser.parse_args()


def _team_labels(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        source, separator, semantic = value.partition("=")
        if not separator or not source.strip() or not semantic.strip():
            raise ValueError(f"无效 --team-label: {value!r}")
        result[source.strip().lower()] = semantic.strip().lower()
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def support_evidence(diagnostic: dict[str, Any], number: int) -> tuple[list[int], list[float]]:
    """Collapse all ROI/augmentations to one strongest observation per source frame."""
    by_frame: dict[int, float] = {}
    for row in diagnostic.get("aggregated_detections", []):
        if int(row.get("number", -1)) != number:
            continue
        frame = int(row["frame"])
        confidence = float(row.get("confidence") or 0.0)
        by_frame[frame] = max(by_frame.get(frame, 0.0), confidence)
    frames = sorted(by_frame)
    return frames, [by_frame[frame] for frame in frames]


def classify_voted(voted: dict[str, Any] | None, diagnostic: dict[str, Any]) -> dict[str, Any]:
    """Boss-compatible three-state policy calibrated to fleeting low-camera views.

    The frozen delivery accepted some two-frame results as confirmed and retained
    strong one-frame results as tentative.  This adapter keeps those semantics,
    while retaining the existing crop-level OCR abstention and close-runner guard.
    """
    if not voted:
        return {
            "status": "unreadable", "number": None, "confidence": 0.0,
            "support_frames": [], "decision_rule": "ocr_abstained",
        }
    number = int(voted["jersey_number"])
    frames, frame_confidences = support_evidence(diagnostic, number)
    support_count = len(frames)
    mean_confidence = sum(frame_confidences) / support_count if support_count else 0.0
    maximum_confidence = max(frame_confidences, default=0.0)
    margin = float(voted.get("winner_margin") or 0.0)
    head_confidence = float(voted.get("head_confidence") or 0.0)
    candidates = voted.get("candidates") or []
    runner = candidates[1] if len(candidates) > 1 else None

    if runner and int(runner.get("votes") or 0) >= 2 and margin < 0.15:
        status = "conflict"
        rule = "competing_numbers_have_independent_frame_support"
    elif support_count >= 2 and mean_confidence >= 0.75 and head_confidence >= 0.60 and margin >= 0.10:
        status = "confirmed"
        rule = "boss_compatible_two_frame_confirmation"
    elif support_count >= 2 and maximum_confidence >= 0.80 and margin >= 0.08:
        status = "tentative"
        rule = "boss_compatible_fleeting_multiframe_candidate"
    elif support_count == 1 and maximum_confidence >= 0.80 and head_confidence >= 0.60 and margin >= 0.08:
        status = "tentative"
        rule = "boss_compatible_single_clear_frame_candidate"
    else:
        status = "unreadable"
        rule = "boss_compatible_evidence_gate_not_met"
    return {
        "status": status, "number": number if status in {"confirmed", "tentative"} else None,
        "confidence": round(mean_confidence, 8), "support_frames": frames,
        "decision_rule": rule, "winner_margin": round(margin, 8),
        "head_confidence": round(head_confidence, 8),
    }


def read_mot_frames(path: Path) -> dict[int, set[int]]:
    frames: dict[int, set[int]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            values = line.split(",")
            frames[int(float(values[1]))].add(int(float(values[0])))
    return dict(frames)


def simultaneous_conflicts(rows: list[dict[str, Any]], frames: dict[int, set[int]]) -> set[int]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        if row["status"] == "confirmed":
            groups[(str(row["team"]), int(row["predicted_number"]))].append(int(row["global_id"]))
    conflicts: set[int] = set()
    for gids in groups.values():
        for index, left in enumerate(gids):
            for right in gids[index + 1:]:
                if frames.get(left, set()) & frames.get(right, set()):
                    conflicts.update((left, right))
    return conflicts


def apply_manual_audit(rows: list[dict[str, Any]], audit: dict[str, Any] | None) -> None:
    """Apply auditable human decisions without silently rewriting OCR evidence."""
    if not audit:
        return
    by_gid = {int(row["global_id"]): row for row in rows}
    allowed = {"confirmed", "tentative", "unreadable", "mismatch", "conflict"}
    for decision in audit.get("decisions", []):
        gid = int(decision["global_id"])
        if gid not in by_gid:
            raise ValueError(f"manual audit global_id missing from OCR results: {gid}")
        status = str(decision["decision"]).strip().lower()
        if status not in allowed:
            raise ValueError(f"invalid manual decision for global_id={gid}: {status}")
        row = by_gid[gid]
        audited_number = decision.get("number")
        if status in {"confirmed", "tentative"}:
            if audited_number is None or int(audited_number) != int(row["predicted_number"]):
                raise ValueError(
                    f"manual audit cannot invent or change OCR number for global_id={gid}"
                )
        else:
            row["predicted_number"] = None
        row["status"] = status
        row["decision_rule"] = f"manual_visual_audit:{decision.get('reason', '').strip()}"


def reclassify(
    *, ocr_dir: Path, mot: Path, output: Path, team_labels: dict[str, str],
    manual_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ocr_dir, mot, output = ocr_dir.resolve(), mot.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"output must not exist: {output}")
    strict_rows = _read_csv(ocr_dir / "jersey_number_results.csv")
    diagnostics = json.loads((ocr_dir / "ocr_diagnostics.json").read_text(encoding="utf-8"))
    tracklets = diagnostics.get("tracklets", {})
    output.mkdir(parents=True)

    results: list[dict[str, Any]] = []
    for source in strict_rows:
        gid = int(source["global_id"])
        diagnostic = tracklets.get(str(gid), tracklets.get(gid, {}))
        decision = classify_voted(diagnostic.get("voted"), diagnostic)
        raw_team = str(source.get("team") or "unknown").strip().lower()
        team = team_labels.get(raw_team, raw_team)
        if team in {"nonplayer", "non_player", "exclude", "excluded"}:
            status, number, rule = "nonplayer", None, "semantic_team_label_nonplayer"
        else:
            status, number, rule = decision["status"], decision["number"], decision["decision_rule"]
        results.append({
            "global_id": gid, "team": team, "predicted_number": number,
            "confidence": decision["confidence"], "status": status,
            "support_frames": ";".join(map(str, decision["support_frames"])),
            "support_count": len(decision["support_frames"]),
            "method": "boss_document_compatible_multiframe_ocr_fusion",
            "decision_rule": rule, "winner_margin": decision.get("winner_margin", 0.0),
            "head_confidence": decision.get("head_confidence", 0.0),
        })

    apply_manual_audit(results, manual_audit)
    collisions = simultaneous_conflicts(results, read_mot_frames(mot))
    for row in results:
        if int(row["global_id"]) in collisions:
            row.update(
                status="conflict", predicted_number=None,
                decision_rule="same_team_same_number_visible_simultaneously",
            )

    columns = [
        "global_id", "team", "predicted_number", "confidence", "status",
        "support_frames", "support_count", "method", "decision_rule",
        "winner_margin", "head_confidence",
    ]
    _write_csv(output / "jersey_number_results.csv", results, columns)

    eligibility: dict[str, Any] = {
        "schema_version": "clip-eligibility-v1",
        "eligible_confirmed": [], "excluded_conflict": [], "excluded_mismatch": [],
        "excluded_unreadable": [], "excluded_nonplayer": [],
        "provenance": {
            "source_ocr": str(ocr_dir),
            "policy": "boss frozen delivery compatible confirmed/tentative/unreadable replay",
            "note": "原始 STEP1-STEP5 源码缺失；本适配只重判现有同场帧级 OCR 证据，不复用其他视频号码。",
        },
    }
    review_rows: list[dict[str, Any]] = []
    for row in results:
        common = {
            "global_id": row["global_id"], "team": row["team"],
            "confidence": row["confidence"], "support_count": row["support_count"],
            "support_frames": row["support_frames"], "source_status": row["status"],
            "decision_rule": row["decision_rule"],
        }
        if row["status"] == "confirmed":
            eligibility["eligible_confirmed"].append({
                **common, "final_number": int(row["predicted_number"]),
                "status": "eligible_confirmed",
            })
        elif row["status"] == "conflict":
            eligibility["excluded_conflict"].append({**common, "predicted_number": None})
        elif row["status"] == "mismatch":
            eligibility["excluded_mismatch"].append({**common, "predicted_number": None})
        elif row["status"] == "nonplayer":
            eligibility["excluded_nonplayer"].append(common)
        else:
            item = {**common, "status": "unreadable"}
            if row["status"] == "tentative":
                item["candidate_number"] = int(row["predicted_number"])
                item["candidate_status"] = "tentative"
                review_rows.append({
                    "global_id": row["global_id"], "team": row["team"],
                    "candidate_number": row["predicted_number"],
                    "confidence": row["confidence"], "support_count": row["support_count"],
                    "support_frames": row["support_frames"], "decision_rule": row["decision_rule"],
                })
            eligibility["excluded_unreadable"].append(item)
    (output / "clip_eligibility.json").write_text(
        json.dumps(eligibility, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    _write_csv(
        output / "jersey_number_review_candidates.csv", review_rows,
        ["global_id", "team", "candidate_number", "confidence", "support_count", "support_frames", "decision_rule"],
    )
    counts = defaultdict(int)
    for row in results:
        counts[str(row["status"])] += 1
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_ocr": str(ocr_dir), "mot": str(mot), "global_ids": len(results),
        "counts": dict(counts), "simultaneous_conflict_global_ids": sorted(collisions),
        "manual_audit_applied": bool(manual_audit),
        "manual_audit_reviewer": (manual_audit or {}).get("reviewer"),
        "thresholds": {
            "confirmed": "support_frames>=2, mean_frame_confidence>=0.75, head>=0.60, margin>=0.10",
            "tentative_multiframe": "support_frames>=2, max_frame_confidence>=0.80, margin>=0.08",
            "tentative_single_frame": "support_frames=1, max_frame_confidence>=0.80, head>=0.60, margin>=0.08",
            "competition_conflict": "runner_votes>=2 and winner_margin<0.15",
        },
    }
    (output / "number_recognition_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = reclassify(
        ocr_dir=args.ocr_dir, mot=args.mot, output=args.output,
        team_labels=_team_labels(args.team_label),
        manual_audit=(json.loads(args.manual_audit.read_text(encoding="utf-8"))
                      if args.manual_audit else None),
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
