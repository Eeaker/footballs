from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_assignments(path: Path, max_players: int) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("assignments", payload)
    result: dict[int, dict] = {}
    for candidate, value in raw.items():
        candidate_id = int(candidate)
        if isinstance(value, int):
            value = {"player_id": value}
        player_id = int(value["player_id"])
        if not 0 <= player_id < max_players:
            raise ValueError(f"player_id必须在0..{max_players - 1}: {player_id}")
        team = value.get("team")
        if team is not None and (not isinstance(team, str) or not team.strip()):
            raise ValueError(f"team必须是非空字符串或null: {team!r}")
        result[candidate_id] = {
            "player_id": player_id,
            "team": team,
            "reviewed": bool(value.get("reviewed", False)),
        }
    return result


def remap_mot(source: Path, target: Path, assignments: dict[int, dict]) -> dict:
    written = skipped = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            fields = line.rstrip().split(",")
            candidate_id = int(fields[1])
            assignment = assignments.get(candidate_id)
            if assignment is None:
                skipped += 1
                continue
            fields[1] = str(assignment["player_id"])
            dst.write(",".join(fields) + "\n")
            written += 1
    return {"written_rows": written, "unassigned_rows": skipped}


def remap_events(source: Path, target: Path, assignments: dict[int, dict]) -> dict:
    events = json.loads(source.read_text(encoding="utf-8"))
    assigned = review = 0
    for event in events:
        candidate = event.get("primary_global_id")
        assignment = assignments.get(int(candidate)) if candidate is not None else None
        event["candidate_global_id"] = candidate
        if assignment is None:
            event["player_id"] = None
            event["team"] = None
            event["identity_review_required"] = True
            review += 1
        else:
            event["player_id"] = assignment["player_id"]
            event["team"] = assignment["team"]
            event["identity_review_required"] = not assignment["reviewed"]
            assigned += 1
            review += int(not assignment["reviewed"])
    target.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"events": len(events), "assigned_events": assigned, "review_events": review}


def main() -> None:
    parser = argparse.ArgumentParser(description="将候选global ID映射为本场16名球员ID")
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--max-players", type=int, default=16)
    args = parser.parse_args()
    assignments = load_assignments(args.mapping, args.max_players)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "max_players": args.max_players,
        "mapped_candidate_ids": len(assignments),
        "mot": remap_mot(args.mot, args.outdir / "tracking_player_mot.txt", assignments),
        "events": remap_events(
            args.events, args.outdir / "event_index_players.json", assignments
        ),
    }
    (args.outdir / "identity_mapping_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
