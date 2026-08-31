from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from mode_split.audit_mot import sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--manual-audit", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    candidates = list(csv.DictReader(args.candidates.open(encoding="utf-8-sig")))
    transitions = defaultdict(list)
    for row in candidates:
        transitions[int(row["global_id"])].append((
            int(row["cut_frame"]), int(row["before_mode"]), int(row["after_mode"]),
        ))
    for gid in transitions:
        transitions[gid].sort()

    data = np.load(args.observations)
    frames, gids, labels = data["frame"], data["gid"], data["team_mode"]
    segment_counts = defaultdict(Counter)
    for frame, gid, label in zip(frames.tolist(), gids.tolist(), labels.tolist()):
        if label < 0:
            continue
        cuts = [row[0] for row in transitions.get(gid, [])]
        segment_index = sum(frame >= cut for cut in cuts)
        segment_counts[(gid, segment_index)][label] += 1
    segment_modes = {}
    for gid, items in transitions.items():
        fallback = items[0][1]
        for segment_index in range(len(items) + 1):
            counts = segment_counts.get((gid, segment_index), Counter())
            segment_modes[(gid, segment_index)] = (
                counts.most_common(1)[0][0] if counts else fallback
            )
            if segment_index < len(items):
                fallback = items[segment_index][2]

    all_gids = set()
    with args.mot.open(encoding="utf-8-sig") as handle:
        for line in handle:
            values = line.split(",")
            all_gids.add(int(float(values[1])))
    next_gid = max(all_gids) + 1
    partition_ids = {}
    for gid in sorted(all_gids):
        if not transitions.get(gid):
            partition_ids[(gid, None)] = gid
            continue
        initial_mode = segment_modes[(gid, 0)]
        partition_ids[(gid, initial_mode)] = gid
        for segment_index in range(len(transitions[gid]) + 1):
            mode = segment_modes[(gid, segment_index)]
            if (gid, mode) not in partition_ids:
                partition_ids[(gid, mode)] = next_gid
                next_gid += 1

    destination = args.output / "tracking_mot_team_partition.txt"
    source_rows = output_rows = duplicate_count = 0
    seen = set()
    with args.mot.open(encoding="utf-8-sig") as source, destination.open("w", encoding="utf-8", newline="") as target:
        for line in source:
            source_rows += 1
            values = line.rstrip("\n").split(",")
            frame, gid = int(float(values[0])), int(float(values[1]))
            if transitions.get(gid):
                cuts = [row[0] for row in transitions[gid]]
                segment_index = sum(frame >= cut for cut in cuts)
                mode = segment_modes[(gid, segment_index)]
                new_gid = partition_ids[(gid, mode)]
            else:
                mode, new_gid = None, partition_ids[(gid, None)]
            values[1] = str(new_gid)
            target.write(",".join(values) + "\n")
            output_rows += 1
            key = frame, new_gid
            if key in seen:
                duplicate_count += 1
            seen.add(key)

    counts = defaultdict(Counter)
    for frame, gid, label in zip(frames.tolist(), gids.tolist(), labels.tolist()):
        if label < 0:
            continue
        if transitions.get(gid):
            cuts = [row[0] for row in transitions[gid]]
            segment_index = sum(frame >= cut for cut in cuts)
            inferred = segment_modes[(gid, segment_index)]
            new_gid = partition_ids[(gid, inferred)]
        else:
            inferred, new_gid = None, gid
        counts[(gid, inferred, new_gid)][label] += 1
    mapping_rows, purities = [], []
    for (gid, mode, new_gid), item in sorted(counts.items(), key=lambda row: row[0][2]):
        purity = max(item.values()) / sum(item.values())
        purities.append(purity)
        mapping_rows.append({
            "original_global_id": gid, "team_mode": "unpartitioned" if mode is None else mode,
            "partition_global_id": new_gid, "mode_0_observations": item[0],
            "mode_1_observations": item[1], "team_mode_purity": round(purity, 8),
        })
    with (args.output / "team_partition_map.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping_rows[0]))
        writer.writeheader(); writer.writerows(mapping_rows)

    audit = list(csv.DictReader(args.manual_audit.open(encoding="utf-8-sig")))
    report = {
        "schema_version": 1,
        "status": "validated_team_partition_for_downstream_full_pipeline",
        "policy": "partition each original global_id by open-set team mode; reuse ID across same-team reappearances",
        "sources": {"mot": str(args.mot.resolve()), "mot_sha256": sha256(args.mot)},
        "manual_validation": {
            "sampled": len(audit),
            "approved_identity_mismatches": sum(row["verdict"] == "true_identity_mismatch" for row in audit),
            "covered_global_ids": len({int(row["global_id"]) for row in audit}),
        },
        "partition": {
            "input_global_ids": len(all_gids), "global_ids_with_cross_team_candidates": len(transitions),
            "candidate_transitions": len(candidates), "output_partition_ids": len(set(partition_ids.values())),
            "source_rows": source_rows, "output_rows": output_rows,
            "row_count_preserved": source_rows == output_rows,
            "same_frame_same_partition_id_duplicates": duplicate_count,
            "minimum_partition_team_purity": min(purities),
            "median_partition_team_purity": float(np.median(purities)),
            "partitions_below_0_8_team_purity": sum(value < .8 for value in purities),
            "output_mot_sha256": sha256(destination),
        },
        "limitations": [
            "Same-team same-colour identity merges are preserved.",
            "Partition IDs are technical identities; OCR/identity mapping remains downstream.",
        ],
    }
    (args.output / "team_partition_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
