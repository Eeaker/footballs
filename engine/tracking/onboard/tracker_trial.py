from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import median

import cv2

from .models import TrialMetrics


def resolve_trial_window(total_seconds: float, requested_start: float = 300.0,
                         requested_duration: float = 120.0) -> tuple[float, float]:
    """优先使用第 5~7 分钟；视频较短时退到末尾等长片段。"""
    if total_seconds <= 0: return 0.0, 0.0
    duration = min(requested_duration, total_seconds)
    start = requested_start if requested_start + duration <= total_seconds else max(0.0, total_seconds - duration)
    return start, duration


def summarize_tracks(name: str, tracker_config: str, frame_counts: list[int], track_lengths: Counter,
                     fps: float) -> TrialMetrics:
    """从逐帧框数与 ID 轨迹长度计算统一 A/B 指标。"""
    frames = len(frame_counts); seconds = frames / max(fps, 1e-9)
    return TrialMetrics(name, tracker_config, frames, seconds,
        round(sum(frame_counts) / max(frames, 1), 4), len(track_lengths),
        round(len(track_lengths) / max(seconds / 60, 1e-9), 3),
        float(median(track_lengths.values())) if track_lengths else 0.0,
        sum(length < 10 for length in track_lengths.values()))


def run_tracker_trial(video_path: str | Path, weights: str | Path, tracker_config: str | Path,
                      name: str, start_sec: float = 300.0, duration_sec: float = 120.0,
                      device: str = "0", imgsz: int = 1280, conf: float = .25) -> TrialMetrics:
    """在指定 120 秒窗口独立运行一次 BoT-SORT 并统计局部 ID。"""
    from ultralytics import YOLO
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"无法读取视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_start, actual_duration = resolve_trial_window(total / fps, start_sec, duration_sec)
    start = int(actual_start * fps); limit = min(int(actual_duration * fps), max(0, total - start))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    model, lengths, counts = YOLO(str(weights)), Counter(), []
    progress_step = max(1, int(round(fps * 10)))
    for offset in range(limit):
        ok, frame = cap.read()
        if not ok: break
        result = model.track(frame, classes=[0], tracker=str(tracker_config), persist=True, conf=conf,
                             iou=.5, imgsz=imgsz, device=device, verbose=False)[0]
        ids = [] if result.boxes is None or result.boxes.id is None else result.boxes.id.int().cpu().tolist()
        counts.append(len(ids)); lengths.update(int(i) for i in ids)
        if (offset + 1) % progress_step == 0:
            print(f"  {name}: {(offset + 1) / fps:.0f}/{limit / fps:.0f}s")
    cap.release()
    return summarize_tracks(name, str(tracker_config), counts, lengths, fps)


def recommend_trial(baseline: TrialMetrics, candidate: TrialMetrics) -> tuple[str, str]:
    """以检出保持为约束，优先更少新 ID 和更长轨迹，避免只追求低 ID 数。"""
    detection_ok = candidate.mean_boxes_per_frame >= baseline.mean_boxes_per_frame * .95
    churn_gain = candidate.new_ids_per_minute <= baseline.new_ids_per_minute * .90
    length_ok = candidate.median_track_length_frames >= baseline.median_track_length_frames * .95
    if detection_ok and churn_gain and length_ok:
        return "candidate", "候选配置保持检出率和轨迹长度，同时每分钟新建 ID 至少降低 10%。"
    return "baseline", "候选配置没有同时满足检出保持、ID 碎裂改善和轨迹长度约束，保留基线更稳妥。"

