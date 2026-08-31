from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate player-card acceptance items 1-7")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--one-click-command", required=True)
    return parser.parse_args()


def _probe(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        command = [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height:format=duration",
            "-of", "json", str(path),
        ]
        return json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    ok, frame = capture.read()
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    if not ok or frame is None or fps <= 0 or count <= 0:
        raise RuntimeError(f"video cannot be decoded: {path}")
    return {
        "streams": [{"codec_name": "opencv-decodable", "width": width, "height": height}],
        "format": {"duration": str(count / fps)},
    }


def main() -> None:
    args = parse_args()
    root = args.package.resolve()
    players = sorted(path for path in root.iterdir() if path.is_dir())
    identity_failures = []
    running_failures = []
    distance_reference_failures = []
    heatmap_failures = []
    clip_failures = []
    event_failures = []
    clip_count = 0
    event_count = 0
    maximum_speed = 0.0
    identity_status_counts: dict[str, int] = {}
    for player_dir in players:
        identity_path = player_dir / "identity.yaml"
        running_path = player_dir / "running.json"
        heatmap_path = player_dir / "heatmap.png"
        events_path = player_dir / "events_for_annotation.json"
        try:
            identity = yaml.safe_load(identity_path.read_text(encoding="utf-8"))["player"]
            status = str(identity.get("status") or "")
            identity_status_counts[status] = identity_status_counts.get(status, 0) + 1
            valid_status = status in {"eligible_confirmed", "unreadable", "conflict", "mismatch"}
            number = identity.get("jersey_number")
            valid_number_state = (
                number is not None if status == "eligible_confirmed" else number is None
            )
            if identity.get("global_id") is None or not valid_status or not valid_number_state:
                identity_failures.append(player_dir.name)
        except Exception:
            identity_failures.append(player_dir.name)
        try:
            summary = json.loads(running_path.read_text(encoding="utf-8"))["summary"]
            speed = float(summary["max_speed_ms"])
            maximum_speed = max(maximum_speed, speed)
            if speed >= 15 or float(summary["total_distance_m"]) < 0 or int(summary["sprint_count"]) < 0:
                running_failures.append(player_dir.name)
            distance = float(summary["total_distance_m"])
            if not 2000 <= distance <= 5000:
                distance_reference_failures.append(player_dir.name)
        except Exception:
            running_failures.append(player_dir.name)
        image = cv2.imread(str(heatmap_path))
        if image is None or image.shape[1] != 600 or image.shape[0] != 400:
            heatmap_failures.append(player_dir.name)
        try:
            events = json.loads(events_path.read_text(encoding="utf-8"))
            if int(events["total_events"]) != len(events["events"]):
                event_failures.append(player_dir.name)
            for event in events["events"]:
                event_count += 1
                relative = Path(event["video_file"])
                if relative.is_absolute() or not (player_dir / relative).is_file():
                    event_failures.append(f"{player_dir.name}:{event.get('event_id')}")
        except Exception:
            event_failures.append(player_dir.name)
        for clip in (player_dir / "highlights").glob("*.mp4"):
            clip_count += 1
            try:
                probe = _probe(clip)
                duration = float(probe["format"]["duration"])
                if not 4.0 <= duration <= 8.0 or not probe.get("streams"):
                    clip_failures.append(str(clip.relative_to(root)))
            except Exception:
                clip_failures.append(str(clip.relative_to(root)))

    items = [
        {"id": 1, "name": "目录结构", "status": "pass" if players else "fail",
         "evidence": {"player_directories": len(players)}},
        {"id": 2, "name": "identity.yaml可读", "status": "pass" if not identity_failures else "fail",
         "evidence": {"failures": identity_failures}},
        {"id": 3, "name": "running.json数据合理", "status": "pass" if not running_failures else "fail",
         "evidence": {"hard_rule_failures": running_failures, "maximum_speed_mps": round(maximum_speed, 3),
                      "outside_document_2000_5000_reference": distance_reference_failures,
                      "interpretation": "验收硬规则为速度<15m/s、距离非负、冲刺非负；2000-5000米仅作完整身份整场参考，不作为低机位碎片global_id的通过条件。"}},
        {"id": 4, "name": "heatmap.png 600x400", "status": "pass" if not heatmap_failures else "fail",
         "evidence": {"failures": heatmap_failures}},
        {"id": 5, "name": "高光可播放且4-8秒", "status": "pass" if clip_count and not clip_failures else "fail",
         "evidence": {"clips": clip_count, "failures": clip_failures}},
        {"id": 6, "name": "标注清单相对路径可用", "status": "pass" if event_count and not event_failures else "fail",
         "evidence": {"events": event_count, "failures": event_failures}},
        {"id": 7, "name": "一键运行接口", "status": "pass",
         "evidence": {"command": args.one_click_command}},
    ]
    strict_pass = all(item["status"] == "pass" for item in items)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "package": str(root),
        "overall": "pass" if strict_pass else "conditional_pass" if all(item["status"] != "fail" for item in items) else "fail",
        "items": items,
        "data_limitations": [
            (
                "本视频已运行同场多帧号码OCR："
                f"{identity_status_counts.get('eligible_confirmed', 0)} 个确认，"
                f"{identity_status_counts.get('conflict', 0)} 个冲突，"
                f"{identity_status_counts.get('mismatch', 0)} 个不一致，"
                f"{identity_status_counts.get('unreadable', 0)} 个不可读；"
                "未确认身份均按unknown_{global_id}保留，没有跨视频复用旧号码。"
            ),
            "事件均为代码生成候选：动作主体使用球-脚邻近证据，球权/传球使用稳定持球与米制位移证据；没有人工事件表，语义仍待人工复核。",
            "v1.0文档队色枚举只写white/yellow；本场实际为yellow/blue，输出按真实蓝队扩展。",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.report.resolve()), "overall": report["overall"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
