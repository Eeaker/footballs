"""Write a no-calibration MOT identity and continuity preflight report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from running_metrics_v1.mot import inspect_records, read_mot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--total-proc-frames", type=int)
    args = parser.parse_args()
    report = inspect_records(read_mot(args.mot), args.total_proc_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "identities"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

