from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATCH_ANALYSIS_ROOT = ROOT / "engine" / "match_analysis"
if str(MATCH_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(MATCH_ANALYSIS_ROOT))

from analysis_lib.io import read_mot  # noqa: E402
from analysis_lib.teams import assign_teams_kmeans  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--mot", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--clusters", type=int, default=3)
    ap.add_argument("--samples", type=int, default=12)
    args = ap.parse_args()
    _, rows = read_mot(args.mot)
    _, diagnostics = assign_teams_kmeans(args.video, rows, args.clusters, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["global_id", "team_id", "samples", "nearest_center_distance", "center_margin", "assignment_method"])
        writer.writeheader(); writer.writerows(diagnostics)
    print(f"team hints: {args.output}")

if __name__ == "__main__":
    main()
