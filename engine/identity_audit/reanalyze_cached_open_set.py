from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mode_split.open_set_team import detect_persistent_team_switches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--parent-report", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--minimum-confident", type=int, default=15)
    ap.add_argument("--purity", type=float, default=.80)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    data = np.load(args.observations)
    frames, gids, labels = data["frame"], data["gid"], data["team_mode"]
    rows = []
    for gid in sorted(np.unique(gids).tolist()):
        mask = gids == gid
        for item in detect_persistent_team_switches(
            frames[mask], labels[mask], window=args.window,
            minimum_confident=args.minimum_confident, purity=args.purity,
        ):
            rows.append({
                "global_id": int(gid), "cut_frame": item.frame,
                "before_mode": item.before_mode, "after_mode": item.after_mode,
                "before_support": item.before_support, "after_support": item.after_support,
            })
    fields = ["global_id", "cut_frame", "before_mode", "after_mode", "before_support", "after_support"]
    with (args.output / "team_switch_candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    parent = json.loads(args.parent_report.read_text(encoding="utf-8"))
    gid11 = [row["cut_frame"] for row in rows if row["global_id"] == 11]
    report = {
        "schema_version": 1,
        "status": "experimental_candidates_require_video_review",
        "parent_audit_report": str(args.parent_report.resolve()),
        "cached_observations": str(args.observations.resolve()),
        "change_from_parent": "enforce alternating two-state transition consistency",
        "parameters": {"window": args.window, "minimum_confident": args.minimum_confident, "purity": args.purity},
        "quality": parent["quality"],
        "model": parent["model"],
        "counts": {
            "input_global_ids": int(len(np.unique(gids))), "candidate_switches": len(rows),
            "ids_with_candidates": len({row["global_id"] for row in rows}),
        },
        "known_regression": {
            "global_id": 11, "expected_switch_window": [28168, 28184],
            "detected_cuts": gid11, "passed": any(28168 <= frame <= 28184 for frame in gid11),
        },
        "safety": parent["safety"],
    }
    (args.output / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
