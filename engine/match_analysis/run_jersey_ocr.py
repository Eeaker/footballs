from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis_lib.jersey_numbers import run_jersey_number_recognition


def main() -> None:
    parser = argparse.ArgumentParser(description="视频+MOT多帧号码识别，输出球员卡号码校验输入")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--team-hints", type=Path, required=True,
                        help="含global_id/team的CSV或clip_eligibility.json")
    parser.add_argument("--output", type=Path, required=True, help="必须不存在")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--maximum-candidates-per-id", type=int, default=36)
    parser.add_argument("--reuse-candidates", type=Path,
                        help="复用此前候选目录，跳过原视频抽帧")
    args = parser.parse_args()
    manifest = run_jersey_number_recognition(
        video=args.video, mot=args.mot, team_hints=args.team_hints, output=args.output,
        gpu=not args.cpu, maximum_candidates_per_id=args.maximum_candidates_per_id,
        reuse_candidates=args.reuse_candidates,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
