from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from mode_split.audit_mot import sha256, write_split_mot


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
    rows = list(csv.DictReader(args.candidates.open(encoding="utf-8-sig")))
    cuts = defaultdict(list)
    for row in rows:
        cuts[int(row["global_id"])].append(int(row["cut_frame"]))
    destination = args.output / "tracking_mot_team_switch_split.txt"
    mapping = write_split_mot(args.mot, destination, dict(cuts))
    with (args.output / "segment_map.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping[0]))
        writer.writeheader(); writer.writerows(mapping)

    source_rows = sum(1 for _ in args.mot.open(encoding="utf-8-sig"))
    output_rows = sum(1 for _ in destination.open(encoding="utf-8"))
    duplicate_count = 0
    seen = set()
    with destination.open(encoding="utf-8") as handle:
        for line in handle:
            v = line.split(",")
            key = int(float(v[0])), int(float(v[1]))
            if key in seen:
                duplicate_count += 1
            seen.add(key)

    observations = np.load(args.observations)
    frames, gids, labels = observations["frame"], observations["gid"], observations["team_mode"]
    segment_mode_counts = defaultdict(Counter)
    for frame, gid, label in zip(frames.tolist(), gids.tolist(), labels.tolist()):
        if label < 0:
            continue
        segment_index = sum(frame >= cut for cut in sorted(cuts.get(gid, [])))
        segment_mode_counts[(gid, segment_index)][label] += 1
    purities = []
    for counts in segment_mode_counts.values():
        purities.append(max(counts.values()) / sum(counts.values()))

    manual_rows = list(csv.DictReader(args.manual_audit.open(encoding="utf-8-sig")))
    approved = sum(row["verdict"] == "true_identity_mismatch" for row in manual_rows)
    report = {
        "schema_version": 1,
        "status": "validated_experimental_mot_for_downstream_full_pipeline",
        "sources": {
            "mot": str(args.mot.resolve()), "mot_sha256": sha256(args.mot),
            "candidates": str(args.candidates.resolve()),
            "observations": str(args.observations.resolve()),
            "manual_audit": str(args.manual_audit.resolve()),
        },
        "manual_validation": {
            "sampled": len(manual_rows), "approved_identity_mismatches": approved,
            "sample_precision": approved / max(1, len(manual_rows)),
            "covered_global_ids": len({int(row["global_id"]) for row in manual_rows}),
        },
        "split": {
            "candidate_cuts": len(rows), "input_global_ids": len(set(gids.tolist())),
            "output_segments": len(mapping), "source_rows": source_rows, "output_rows": output_rows,
            "row_count_preserved": source_rows == output_rows,
            "same_frame_same_segment_id_duplicates": duplicate_count,
            "minimum_segment_team_purity": min(purities) if purities else None,
            "median_segment_team_purity": float(np.median(purities)) if purities else None,
            "segments_below_0_8_team_purity": sum(value < .8 for value in purities),
            "output_mot_sha256": sha256(destination),
        },
        "limitations": [
            "Split segments are new technical identities, not confirmed jersey-number identities.",
            "Same-team same-colour identity merges are outside this detector's scope.",
        ],
    }
    (args.output / "split_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
