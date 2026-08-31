from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2

from analysis_lib.player_card import generate_player_card_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从视频 + MOT + 号码 + 事件生成完整球员卡数据包")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--mot", type=Path,
        help="此前追踪生成的 tracking_mot.txt；提供预计算跑动时序时仍用于交付包溯源和未识别球员片段",
    )
    parser.add_argument("--numbers", type=Path,
                        help="号码结果：clip_eligibility.json 或文档v1.0的jersey_number_results.csv")
    parser.add_argument("--run-number-ocr", action="store_true",
                        help="直接从视频+MOT运行迁移的多帧号码识别，再继续生成球员卡")
    parser.add_argument("--team-hints", type=Path,
                        help="global_id到队伍的CSV或clip_eligibility.json；OCR或无team列CSV需要")
    parser.add_argument("--number-ocr-output", type=Path,
                        help="号码识别独立输出目录；默认在球员卡输出旁创建 *_number_ocr")
    parser.add_argument("--ocr-cpu", action="store_true")
    parser.add_argument("--ocr-maximum-candidates-per-id", type=int, default=36)
    parser.add_argument("--ocr-reuse-candidates", type=Path)
    parser.add_argument("--events", type=Path, required=True,
                        help="标准 events_for_annotation.json")
    parser.add_argument("--calibration", type=Path,
                        help="可选；不传时按视频尺寸/FPS/帧数自动发现同场动态标定")
    parser.add_argument("--running-timeseries", type=Path,
                        help="旧接口兼容：直接提供已经生成的米制跑动时序")
    parser.add_argument("--fps", type=float, help="旧接口兼容；MOT 模式从视频/标定读取")
    parser.add_argument(
        "--running-src", type=Path,
        default=Path(__file__).resolve().parent.parent / "football_metric_running" / "src",
        help="football_metric_running/src；直接调用其原始跑动模块",
    )
    parser.add_argument("--output", type=Path, required=True, help="必须不存在，避免覆盖既有数据")
    parser.add_argument(
        "--formal-output", type=Path,
        help="可选；同一条命令继续导出对接文档v1.0目录（也必须不存在）",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, float | int]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"视频不存在: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    metadata = {
        "frame_width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "frame_height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "total_frames": int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
    }
    cap.release()
    if metadata["fps"] <= 0 or metadata["total_frames"] <= 0:
        raise RuntimeError(f"视频元数据无效: {metadata}")
    return metadata


def _calibration_matches(path: Path, video_meta: dict[str, float | int]) -> tuple[bool, bool]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data["video_metadata"]
        passed = bool(data.get("validation", {}).get("passed", False))
        dimensions_match = (
            int(meta["frame_width"]) == int(video_meta["frame_width"])
            and int(meta["frame_height"]) == int(video_meta["frame_height"])
        )
        fps_match = abs(float(meta["proc_fps"]) - float(video_meta["fps"])) <= 0.02
        frames_match = abs(int(meta["proc_total_frames"]) - int(video_meta["total_frames"])) <= 2
        usable = passed and dimensions_match and fps_match and frames_match and "field_bounds_m" in data
        dynamic = data.get("camera_model") == "dynamic_per_frame_homography"
        return usable, dynamic
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False, False


def resolve_calibration(video: Path, mot: Path, explicit: Path | None = None) -> Path:
    video_meta = probe_video(video)
    if explicit is not None:
        explicit = explicit.resolve()
        usable, _ = _calibration_matches(explicit, video_meta)
        if not usable:
            raise ValueError(f"标定未通过或与视频尺寸/FPS/帧数不匹配: {explicit}")
        return explicit

    local_names = (
        "dynamic_calibration.json", "dynamic_calibration_45x25.json",
        "u12_dynamic_45x25_full.json", "calibration.json",
    )
    candidates: list[Path] = []
    for directory in (mot.resolve().parent, video.resolve().parent):
        candidates.extend(directory / name for name in local_names)

    workspace = Path(__file__).resolve().parent.parent
    candidates.extend(workspace.rglob("*calibration*.json"))
    candidates.extend(workspace.rglob("*dynamic*.json"))
    compatible: dict[str, tuple[Path, bool]] = {}
    for candidate in candidates:
        if not candidate.is_file():
            continue
        usable, dynamic = _calibration_matches(candidate, video_meta)
        if usable:
            compatible.setdefault(_sha256(candidate), (candidate.resolve(), dynamic))
    if not compatible:
        raise FileNotFoundError(
            "没有发现与视频严格匹配且验证通过的标定；请用 --calibration 指定同场标定"
        )
    dynamic = sorted(path for path, is_dynamic in compatible.values() if is_dynamic)
    if len(dynamic) == 1:
        return dynamic[0]
    if len(dynamic) > 1:
        raise RuntimeError(f"发现多个内容不同的匹配动态标定，请显式指定 --calibration: {dynamic}")
    static = sorted(path for path, _ in compatible.values())
    if len(static) == 1:
        return static[0]
    raise RuntimeError(f"发现多个内容不同的匹配标定，请显式指定 --calibration: {static}")


def calculate_running_from_mot(
    *, mot: Path, calibration: Path, outdir: Path, running_src: Path,
) -> tuple[Path, Path]:
    mot = mot.resolve()
    if not mot.is_file():
        raise FileNotFoundError(f"MOT 不存在: {mot}")
    module = running_src.resolve() / "running_metrics_v1" / "calculate_running.py"
    if not module.is_file():
        raise FileNotFoundError(f"米制跑动源码不存在: {module}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(running_src.resolve()) + os.pathsep + environment.get("PYTHONPATH", "")
    command = [
        sys.executable, "-m", "running_metrics_v1.calculate_running",
        "--mot", str(mot), "--calibration", str(calibration.resolve()), "--outdir", str(outdir),
    ]
    subprocess.run(command, check=True, env=environment)
    timeseries = outdir / "player_running_timeseries.csv"
    quality = outdir / "running_quality_report.json"
    if not timeseries.is_file() or not quality.is_file():
        raise RuntimeError("米制跑动模块未生成预期时序或质量报告")
    return timeseries, quality


def generate_from_mot(
    *, video: Path, mot: Path, numbers: Path, events: Path, output: Path,
    calibration: Path | None = None, running_src: Path,
) -> dict:
    if output.resolve().exists():
        raise FileExistsError(f"output must not exist: {output.resolve()}")
    selected_calibration = resolve_calibration(video, mot, calibration)
    video_meta = probe_video(video)
    output_parent = output.resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.mot-prep-", dir=output_parent) as temp:
        prep = Path(temp)
        timeseries, quality = calculate_running_from_mot(
            mot=mot, calibration=selected_calibration, outdir=prep, running_src=running_src,
        )
        return generate_player_card_data(
            video=video, numbers=numbers, events=events,
            running_timeseries=timeseries, calibration=selected_calibration,
            output=output, fps=float(video_meta["fps"]), source_mot=mot,
            running_quality=quality,
        )


def main() -> None:
    args = parse_args()
    if bool(args.numbers) == bool(args.run_number_ocr):
        raise ValueError("--numbers 与 --run-number-ocr 必须且只能选择一个")
    numbers = args.numbers
    if args.run_number_ocr:
        if not args.mot or not args.team_hints:
            raise ValueError("--run-number-ocr 需要 --mot 和 --team-hints")
        from analysis_lib.jersey_numbers import run_jersey_number_recognition
        ocr_output = args.number_ocr_output or args.output.with_name(args.output.name + "_number_ocr")
        run_jersey_number_recognition(
            video=args.video, mot=args.mot, team_hints=args.team_hints, output=ocr_output,
            gpu=not args.ocr_cpu,
            maximum_candidates_per_id=args.ocr_maximum_candidates_per_id,
            reuse_candidates=args.ocr_reuse_candidates,
        )
        numbers = ocr_output / "clip_eligibility.json"
    elif numbers and numbers.suffix.lower() == ".csv":
        from analysis_lib.jersey_numbers import adapt_number_results_csv
        adapter_output = args.output.with_name(args.output.name + "_number_adapter") / "clip_eligibility.json"
        numbers = adapt_number_results_csv(
            numbers=numbers, mot=args.mot, team_hints=args.team_hints, output=adapter_output,
        )
    if args.running_timeseries:
        if not args.calibration or args.fps is None:
            raise ValueError("提供 --running-timeseries 时还必须同时提供 --calibration --fps")
        manifest = generate_player_card_data(
            video=args.video,
            numbers=numbers,
            events=args.events,
            running_timeseries=args.running_timeseries,
            calibration=args.calibration,
            output=args.output,
            fps=args.fps,
            source_mot=args.mot,
        )
    elif args.mot:
        manifest = generate_from_mot(
            video=args.video, mot=args.mot, numbers=numbers, events=args.events,
            output=args.output, calibration=args.calibration, running_src=args.running_src,
        )
    else:
        raise ValueError("请提供 --mot，或同时提供 --running-timeseries --calibration --fps")
    result = {
        "output": str(args.output.resolve()),
        "players": manifest["players"],
        "event_count": manifest["event_count"],
    }
    if args.formal_output:
        from export_player_card_delivery_v1 import export_delivery
        export_delivery(args.output, args.formal_output)
        result["formal_output"] = str(args.formal_output.resolve())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
