from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .passes import PassEvent


def systematic_sample(events: list[PassEvent], size: int) -> list[PassEvent]:
    """Deterministic time-spread sample; no cherry-picking and no duplicate events."""
    if len(events) <= size:
        return list(events)
    indices = [int((index + .5) * len(events) / size) for index in range(size)]
    return [events[min(index, len(events) - 1)] for index in indices]


def make_sample_rows(events: list[PassEvent], size: int, fps: float) -> list[dict]:
    rows = []
    for event in systematic_sample(events, size):
        rows.append({
            "pass_id": event.pass_id,
            "event_time_seconds": round(event.receive_frame_proc / fps, 3),
            "from_global_id": event.from_global_id, "to_global_id": event.to_global_id,
            "from_team_id": event.team_id, "to_team_id": event.team_id,
            "model_outcome": event.classification, "distance_m": round(event.distance_m, 3),
            "human_is_pass": "", "human_outcome": "", "human_note": "",
        })
    return rows


def evaluate_annotations(
    sample_rows: list[dict], annotations: str | Path | None, threshold: float, required_count: int,
) -> dict:
    available = len(sample_rows)
    if annotations is None:
        return {
            "status": "pending_human_review", "required_labels": required_count,
            "available_sample_events": available, "valid_labels": 0,
            "agreement_count": 0, "agreement_rate": None, "threshold": threshold, "passed": False,
        }
    labels = {}
    with Path(annotations).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            value = str(row.get("human_is_pass", "")).strip().lower()
            if value in {"1", "true", "yes", "y", "是"}:
                labels[int(row["pass_id"])] = True
            elif value in {"0", "false", "no", "n", "否"}:
                labels[int(row["pass_id"])] = False
    sampled_ids = {int(row["pass_id"]) for row in sample_rows}
    valid = {key: value for key, value in labels.items() if key in sampled_ids}
    agreement = sum(1 for value in valid.values() if value)
    rate = agreement / len(valid) if valid else None
    passed = len(valid) == required_count and available == required_count and rate is not None and rate >= threshold
    return {
        "status": "passed" if passed else "failed", "available_sample_events": available,
        "required_labels": required_count, "valid_labels": len(valid),
        "agreement_count": agreement, "agreement_rate": rate,
        "threshold": threshold, "passed": passed,
    }


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
