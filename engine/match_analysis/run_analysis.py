from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis_lib.pipeline import PipelineConfig, run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="match analysis 球权、传球网络与一键验收机器人")
    parser.add_argument("--tracking-dir", type=Path, required=True, help="tracking追踪结果目录")
    parser.add_argument("--calibration", type=Path, required=True, help="已独立验证的YAML/JSON米制标定")
    parser.add_argument("--output", type=Path, required=True, help="必须不存在的新输出目录")
    parser.add_argument("--video", type=Path, help="未提供冻结队伍映射时，用于轨迹级球衣K-means")
    parser.add_argument("--team-map", type=Path, help="可选冻结映射CSV: global_id,team_id")
    parser.add_argument("--annotations", type=Path, help="已人工填写的acceptance_sample_20.csv")
    parser.add_argument("--fps", type=float, help="元数据缺失时显式提供processed_fps")
    parser.add_argument("--vid-stride", type=int, help="显式提供追踪抽帧步长")
    parser.add_argument("--team-clusters", type=int, default=2, choices=(2, 3))
    parser.add_argument("--team-samples-per-id", type=int, default=12)
    parser.add_argument("--max-ball-gap-frames", type=int, default=2)
    parser.add_argument("--max-transfer-gap-seconds", type=float, default=1.5)
    parser.add_argument("--min-pass-displacement-m", type=float, default=.5,
                        help="主动定向传递的最小米制净位移；默认0.5m")
    parser.add_argument("--pass-review-min-displacement-m", type=float, default=.25,
                        help="仅供人工复核的灰区下限；默认0.25m，不进入正式传球网络")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        team_clusters=args.team_clusters, team_samples_per_id=args.team_samples_per_id,
        max_ball_gap_frames=args.max_ball_gap_frames,
        max_transfer_gap_seconds=args.max_transfer_gap_seconds,
        min_pass_displacement_m=args.min_pass_displacement_m,
        pass_review_min_displacement_m=args.pass_review_min_displacement_m,
    )
    report = run_analysis(
        tracking_dir=args.tracking_dir, calibration=args.calibration, output=args.output,
        video=args.video, team_map_path=args.team_map, annotations=args.annotations,
        config=config, fps_override=args.fps, vid_stride_override=args.vid_stride,
    )
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
