from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition-map", type=Path, required=True)
    ap.add_argument("--original-team-map", type=Path, required=True)
    ap.add_argument("--mot", type=Path, help="ensure every MOT identity has a frozen team row")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original = {
        int(row["global_id"]): row["team_id"]
        for row in csv.DictReader(args.original_team_map.open(encoding="utf-8-sig"))
    }
    rows = []
    for row in csv.DictReader(args.partition_map.open(encoding="utf-8-sig")):
        source_gid = int(row["original_global_id"])
        mode = row["team_mode"]
        if mode == "0":
            team_id, method = "team_2", "validated_open_set_yellow_mode"
        elif mode == "1":
            team_id, method = "team_1", "validated_open_set_blue_mode"
        else:
            team_id, method = original[source_gid], "preserved_original_unpartitioned_mapping"
        rows.append({
            "global_id": int(row["partition_global_id"]), "team_id": team_id,
            "samples": int(row["mode_0_observations"]) + int(row["mode_1_observations"]),
            "nearest_center_distance": "", "center_margin": row["team_mode_purity"],
            "assignment_method": method,
        })
    if args.mot:
        mot_ids = set()
        with args.mot.open(encoding="utf-8-sig") as handle:
            for line in handle:
                mot_ids.add(int(float(line.split(",")[1])))
        mapped = {row["global_id"] for row in rows}
        for gid in sorted(mot_ids - mapped):
            if gid not in original:
                raise ValueError(f"MOT identity {gid} has no partition or original team mapping")
            rows.append({
                "global_id": gid, "team_id": original[gid], "samples": 0,
                "nearest_center_distance": "", "center_margin": "",
                "assignment_method": "preserved_original_no_accepted_colour_observations",
            })
        rows.sort(key=lambda row: row["global_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
