from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis_lib.pitch_render import render_pitch_video


def main() -> None:
    parser = argparse.ArgumentParser(description="渲染独立米制球场平面：跑动+球权+转换+主动传球网络")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--timeseries", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--start-proc", type=int)
    parser.add_argument("--end-proc", type=int)
    args = parser.parse_args()
    result = render_pitch_video(
        calibration_path=args.calibration,
        timeseries_path=args.timeseries,
        possession_path=args.analysis_dir / "possession_frame_evidence.csv",
        transitions_path=args.analysis_dir / "possession_transitions.csv",
        passes_path=args.analysis_dir / "pass_events.csv",
        team_map_path=args.analysis_dir / "player_team_map.csv",
        output_path=args.output, width=args.width, height=args.height,
        start_proc=args.start_proc, end_proc=args.end_proc,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
