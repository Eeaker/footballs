from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REVIEW_FIELDS = [
    "event_id", "event_time", "base_event_type", "candidate_global_id",
    "actor_attribution_status", "actor_attribution_reason", "id_focus_clip",
    "candidate_labels", "confirmed_global_id", "confirmed_dimension",
    "confirmed_labels", "include_in_highlight", "review_status", "review_notes",
]


def build_rows(events: list[dict], clips: list[dict]) -> list[dict]:
    clip_by_event = {int(row["event_id"]): row["clip_file"] for row in clips}
    rows = []
    for event in events:
        event_id = int(event["event_id"])
        rows.append({
            "event_id": event_id,
            "event_time": event.get("event_time"),
            "base_event_type": event.get("base_event_type"),
            "candidate_global_id": event.get("primary_global_id"),
            "actor_attribution_status": event.get("actor_attribution_status"),
            "actor_attribution_reason": event.get("actor_attribution_reason"),
            "id_focus_clip": clip_by_event.get(event_id),
            "candidate_labels": [],
            "confirmed_global_id": None,
            "confirmed_dimension": None,
            "confirmed_labels": [],
            "include_in_highlight": None,
            "review_status": "pending",
            "review_notes": "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="生成候选事件人工复核清单")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    events = json.loads(args.events.read_text(encoding="utf-8-sig"))
    clips = json.loads(args.clips.read_text(encoding="utf-8-sig"))
    catalog = json.loads(args.labels.read_text(encoding="utf-8-sig"))
    rows = build_rows(events, clips)
    args.outdir.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "pending_human_review",
        "evaluation_policy": "closed_candidate_labels_plus_human_review_no_scoring",
        "label_catalog": {
            "version": catalog["version"],
            "dimensions": catalog["dimensions"] + [catalog["fallback_dimension"]],
            "allowed_labels": catalog["labels"],
        },
        "event_count": len(rows),
        "events": rows,
    }
    (args.outdir / "events_for_human_review.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.outdir / "events_for_human_review.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["candidate_labels"] = "|".join(row["candidate_labels"])
            flat["confirmed_labels"] = "|".join(row["confirmed_labels"])
            writer.writerow(flat)
    print(json.dumps({"review_events": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
