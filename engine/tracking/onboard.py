from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from onboard.calibration import create_keyframe, load_imported_calibration
from onboard.config_builder import (build_pipeline_config, candidate_tracker_config,
                                    load_tracker_yaml, safe_venue_name, write_yaml)
from onboard.field_setup import capture_pitch_polygons, load_field_geometry
from onboard.models import CalibrationResult
from onboard.report import write_report
from onboard.team_colors import analyze_team_colors, select_color_clustering
from onboard.tracker_trial import recommend_trial, run_tracker_trial
from onboard.ui import annotate_video_keyframes, read_frame, select_points
from onboard.video_health import inspect_video_health


ROOT = Path(__file__).resolve().parent


def parse_xy(text: str) -> list[float]:
    """解析人工输入的 X,Y 米制坐标。"""
    parts = text.replace("，", ",").split(",")
    if len(parts) != 2: raise ValueError("格式应为 X,Y，例如 0,0")
    return [float(parts[0]), float(parts[1])]


def confirm_value(label: str, suggested, cast):
    """交互确认一个自动建议值；直接回车也会被记录为人工确认。"""
    answer = input(f"{label} [建议 {suggested}]: ").strip()
    return cast(answer) if answer else cast(suggested)


def run_stage_c(video: Path, mode: str, duration_frames: int, threshold: float = .5) -> CalibrationResult:
    """执行交互标定；验证线段超阈值时要求重标或明确放弃。"""
    suggested = [0] if mode == "static" else [0, duration_frames // 2, max(0, duration_frames - 1)]
    if mode == "static":
        frame = read_frame(video, suggested[0])
        points = select_points(frame, f"Metric calibration frame {suggested[0]}", 4, 8)
        annotations = ([{"frame_index": suggested[0], "points": points}] if points else [])
    else:
        annotations = annotate_video_keyframes(
            video, suggested, title="Dynamic metric reference-point calibration",
            minimum=4, maximum=8, close_shape=False, minimum_keyframes=2,
        )
    if annotations:
        print(f"图像参照点已确认：{len(annotations)} 个关键帧。请回到终端输入各点真实米制坐标。")
    keyframes = []
    for annotation in annotations:
        frame_index = int(annotation["frame_index"])
        points = annotation["points"]
        while True:
            frame = read_frame(video, frame_index)
            if not points: break
            world = []
            print("坐标约定：左下角为原点，X 沿场地长度，Y 沿宽度，单位米。")
            try:
                for i in range(len(points)):
                    world.append(parse_xy(input(f"点 {i + 1} 的真实坐标 X,Y(m): ")))
                validation_points = select_points(frame, "Independent validation segment", 2, 2)
                if len(validation_points) != 2: break
                length = float(input("该独立线段真实长度(m): "))
                item = create_keyframe(frame_index, points, world,
                    [{"p1": validation_points[0], "p2": validation_points[1], "length_m": length}])
            except ValueError as exc:
                print(f"输入或标定失败：{exc}，请重试。")
                continue
            print(f"拟合 RMSE={item.fit_rmse_m:.3f}m；独立验证误差={item.validation_error_m:.3f}m")
            if item.validation_error_m <= threshold:
                keyframes.append(item); break
            print(f"误差超过 {threshold:.2f}m，必须重新点选该帧参照点；按 Esc 可放弃。")
            points = select_points(frame, f"Recalibrate frame {frame_index}", 4, 8)
    validated = bool(keyframes) and all(item.validation_error_m <= threshold for item in keyframes)
    return CalibrationResult(validated, mode, keyframes=keyframes,
                             validation_threshold_m=threshold, validated=validated)


def make_parser() -> argparse.ArgumentParser:
    """构建命令行解析器，使各阶段既可交互也可用于服务器无界面运行。"""
    parser = argparse.ArgumentParser(description="tracking 新视频上机适配工具")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--venue", required=True, help="场地名，用于输出目录和配置文件名")
    parser.add_argument("--weights", type=Path, default=ROOT.parent / "models" / "yolov8x.pt")
    parser.add_argument("--baseline-tracker", type=Path, default=ROOT / "config" / "botsort_buffer.yaml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "onboard_outputs")
    parser.add_argument("--device", default="0")
    parser.add_argument("--expected-players", type=int)
    referee = parser.add_mutually_exclusive_group()
    referee.add_argument("--referee-present", dest="referee_present", action="store_true")
    referee.add_argument("--no-referee", dest="referee_present", action="store_false")
    parser.set_defaults(referee_present=None)
    parser.add_argument("--team-clusters", choices=["auto", "3"], default="auto",
                        help="新视频适配固定自动使用三簇；K=2仅供旧V3配置兼容")
    parser.add_argument("--trial-start", type=float, default=300.0)
    parser.add_argument("--trial-duration", type=float, default=120.0)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--headless", action="store_true",
                        help="保留终端人工确认，但不弹出OpenCV窗口；标定需导入JSON")
    parser.add_argument("--skip-team-colors", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--calibration-mode",
                        choices=["auto", "static", "dynamic_keyframes", "manual_keyframes", "disabled"],
                        default="auto")
    parser.add_argument("--skip-trial", action="store_true")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--min-turf-score", type=float)
    parser.add_argument("--min-track-turf-ratio", type=float)
    parser.add_argument("--min-foot-y-ratio", type=float)
    parser.add_argument("--event-percentile", type=float)
    parser.add_argument("--event-min-gap", type=float)
    parser.add_argument("--edge-margin", type=float)
    parser.add_argument("--event-count", type=int)
    parser.add_argument("--pre-sec", type=float)
    parser.add_argument("--post-sec", type=float)
    parser.add_argument("--track-buffer", type=int)
    parser.add_argument("--match-thresh", type=float)
    parser.add_argument("--vid-stride", type=int)
    parser.add_argument("--min-track-frames", type=int)
    parser.add_argument("--min-presence-ratio", type=float)
    parser.add_argument("--min-geometry-ratio", type=float)
    parser.add_argument("--field-geometry-json", type=Path)
    parser.add_argument("--skip-field-geometry", action="store_true")
    parser.add_argument("--field-margin", type=float, default=12.0,
                        help="图像多边形边界容差，单位像素")
    field = parser.add_mutually_exclusive_group()
    field.add_argument("--field-filter", dest="field_filter_enabled", action="store_true")
    field.add_argument("--disable-field-filter", dest="field_filter_enabled", action="store_false")
    parser.set_defaults(field_filter_enabled=None)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.non_interactive and args.expected_players is None:
        raise SystemExit("无界面模式必须显式提供 --expected-players，避免套用其他视频先验")
    if args.non_interactive and args.referee_present is None:
        raise SystemExit("无界面模式必须显式提供 --referee-present 或 --no-referee")
    if args.non_interactive and args.field_filter_enabled is None:
        raise SystemExit("无界面模式必须显式提供 --field-filter 或 --disable-field-filter")
    expected_players = args.expected_players
    if not args.non_interactive and expected_players is None:
        expected_players = confirm_value("场上预期球员人数（含守门员）", 16, int)
    if not args.non_interactive and args.referee_present is None:
        args.referee_present = input("场上是否有裁判 [y/N]: ").strip().lower() in {"y", "yes", "1"}
    if not args.non_interactive and args.field_filter_enabled is None:
        args.field_filter_enabled = input("是否启用草地/脚点场地过滤 [Y/n]: ").strip().lower() not in {"n", "no", "0"}
    if expected_players is None or expected_players < 1 or args.trial_start < 0 or args.trial_duration <= 0:
        raise SystemExit("人数和试跑时间参数非法")
    venue = safe_venue_name(args.venue); output = args.output_root / venue; output.mkdir(parents=True, exist_ok=True)
    suggestions = {
        "confidence": .25, "imgsz": 1280, "min_turf_score": .15,
        "min_track_turf_ratio": .25, "min_geometry_ratio": .60, "min_foot_y_ratio": .32,
        "event_percentile": 92.0, "event_min_gap": 2.0, "edge_margin": 20.0,
        "event_count": 5, "pre_sec": 15.0, "post_sec": 15.0, "vid_stride": 1,
        "min_track_frames": 10, "min_presence_ratio": .005,
    }
    option_names = {
        "confidence": "conf", "imgsz": "imgsz", "min_turf_score": "min_turf_score",
        "min_track_turf_ratio": "min_track_turf_ratio", "min_geometry_ratio": "min_geometry_ratio",
        "min_foot_y_ratio": "min_foot_y_ratio", "event_percentile": "event_percentile",
        "event_min_gap": "event_min_gap", "edge_margin": "edge_margin", "event_count": "event_count",
        "pre_sec": "pre_sec", "post_sec": "post_sec", "min_track_frames": "min_track_frames",
        "min_presence_ratio": "min_presence_ratio", "vid_stride": "vid_stride",
    }
    manual = {}
    for key, suggested in suggestions.items():
        supplied = getattr(args, option_names[key])
        if supplied is not None:
            manual[key] = supplied
        elif args.non_interactive:
            raise SystemExit(f"无界面模式必须显式提供 --{option_names[key].replace('_', '-')}")
        else:
            manual[key] = confirm_value(key, suggested, int if isinstance(suggested, int) else float)
    manual["field_filter_enabled"] = bool(args.field_filter_enabled)

    print("[Stage A] 视频体检与稀疏光流相机运动分析")
    health = inspect_video_health(args.video)
    (output / "stage_a_health.json").write_text(json.dumps(health.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"相机={health.motion_type}；标定建议={health.calibration_mode}；{health.recommendation}")

    print("[Stage B] 场地几何与比赛区域")
    if args.field_geometry_json:
        field_geometry = load_field_geometry(args.field_geometry_json)
    elif (not args.field_filter_enabled or args.skip_field_geometry
          or args.headless or args.non_interactive):
        field_geometry = {"enabled": False, "mode": "disabled",
                          "reason": "not_configured_or_no_gui"}
    else:
        print("请沿可见球场边界同方向点击4–8点；动态镜头允许各关键帧使用不同点数。")
        field_geometry = capture_pitch_polygons(
            args.video, health.metadata.frame_count,
            dynamic=health.motion_type != "fixed", margin_px=args.field_margin,
        )
    (output / "stage_b_field_geometry.json").write_text(
        json.dumps(field_geometry, ensure_ascii=False, indent=2), encoding="utf-8")

    colors = None
    if not args.skip_team_colors:
        print("[Stage C] 短连续片段跟踪与自动三簇外观建模")
        colors, analysis = analyze_team_colors(
            args.video, args.weights, output / "team_color_board.jpg", args.device,
            min_turf_support=float(manual["min_turf_score"]),
            min_track_turf_ratio=float(manual["min_track_turf_ratio"]),
            min_geometry_ratio=float(manual["min_geometry_ratio"]),
            field_geometry=field_geometry, tracker_config=args.baseline_tracker,
        )
        selected = 3
        print(f"色板已自动导出：{colors.board_path}；诊断轮廓系数 K2={colors.silhouette_k2:.3f}, K3={colors.silhouette_k3:.3f}")
        colors = select_color_clustering(colors, analysis, selected, output / "team_color_board.jpg")
        color_payload = {**colors.to_dict(), "track_filter": analysis.get("track_filter", [])}
        (output / "stage_c_colors.json").write_text(json.dumps(color_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Stage D] 可选米制标定")
    calibration_mode = args.calibration_mode
    if args.headless and not args.calibration_json and not args.skip_calibration:
        print("当前为无图形界面模式：没有导入 --calibration-json，本轮关闭米制标定；可在本地GUI完成后重新导入。")
        calibration_mode = "disabled"
    if not args.non_interactive and calibration_mode == "auto" and not args.skip_calibration:
        answer = input(f"标定模式 [建议 {health.calibration_mode}; static/dynamic_keyframes/manual_keyframes/disabled]: ").strip()
        calibration_mode = answer or health.calibration_mode
    if args.calibration_json:
        calibration = load_imported_calibration(args.calibration_json)
    elif args.skip_calibration or calibration_mode == "disabled" or args.non_interactive:
        calibration = CalibrationResult(False, "disabled", validation_threshold_m=.5, validated=False)
    else:
        calibration = run_stage_c(args.video, calibration_mode, health.metadata.frame_count)
    (output / "stage_d_metric_calibration.json").write_text(json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_yaml = load_tracker_yaml(args.baseline_tracker)
    candidate_yaml = candidate_tracker_config(baseline_yaml, health)
    if args.non_interactive:
        if args.track_buffer is not None: candidate_yaml["track_buffer"] = args.track_buffer
        if args.match_thresh is not None: candidate_yaml["match_thresh"] = args.match_thresh
    else:
        candidate_yaml["track_buffer"] = confirm_value("候选 track_buffer（帧）", candidate_yaml["track_buffer"], int)
        candidate_yaml["match_thresh"] = confirm_value("候选 match_thresh", candidate_yaml["match_thresh"], float)
    baseline_path = write_yaml(output / "botsort_baseline.yaml", baseline_yaml)
    candidate_path = write_yaml(output / "botsort_candidate.yaml", candidate_yaml)
    baseline = candidate = None; selected_name = "baseline"; reason = "Stage D 未执行，保留已经验证的 V3 基线。"
    if not args.skip_trial:
        print("[Stage E] 同一 120 秒片段 A/B 试跑（将执行两遍）")
        baseline = run_tracker_trial(args.video, args.weights, baseline_path, "baseline", args.trial_start,
                                     args.trial_duration, args.device)
        candidate = run_tracker_trial(args.video, args.weights, candidate_path, "candidate", args.trial_start,
                                      args.trial_duration, args.device)
        selected_name, reason = recommend_trial(baseline, candidate)
    selected_path = baseline_path if selected_name == "baseline" else candidate_path
    if colors:
        manual["team_clusters"] = colors.selected_k
    elif args.team_clusters == "3":
        manual["team_clusters"] = int(args.team_clusters)
    else:
        manual["team_clusters"] = confirm_value("球衣/角色颜色簇数", 2, int)
    config = build_pipeline_config(venue, args.video, health, colors, calibration, selected_path,
                                   baseline if selected_name == "baseline" else candidate,
                                   expected_players, args.referee_present, args.weights, manual,
                                   field_geometry)
    config_path = write_yaml(output / f"config_{venue}.yaml", config)
    report_path = write_report(output / "adaptation_report.md", health, colors, calibration,
                               baseline, candidate, selected_name, reason)
    print(f"完成：\n配置 {config_path.resolve()}\n报告 {report_path.resolve()}")


if __name__ == "__main__":
    main()
