from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import MotionHealth, VideoMetadata


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    """读取视频帧率、尺寸、帧数和时长，并验证文件可解码。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法读取视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoMetadata(str(Path(video_path).resolve()), fps, width, height, frames, frames / fps)


def uniform_frame_indices(frame_count: int, count: int, margin_ratio: float = 0.02) -> list[int]:
    """在去除首尾边缘后均匀产生不重复帧号。"""
    if frame_count <= 0 or count <= 0:
        return []
    lo = min(frame_count - 1, max(0, int(frame_count * margin_ratio)))
    hi = max(lo, min(frame_count - 1, int(frame_count * (1 - margin_ratio)) - 1))
    return sorted(set(np.linspace(lo, hi, min(count, hi - lo + 1)).round().astype(int).tolist()))


def estimate_pair_motion(previous: np.ndarray, current: np.ndarray) -> dict | None:
    """以稀疏光流和稳健仿射拟合估计两帧背景相机运动。"""
    prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY) if previous.ndim == 3 else previous
    curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) if current.ndim == 3 else current
    points = cv2.goodFeaturesToTrack(prev_gray, 600, 0.01, 8, blockSize=5)
    if points is None or len(points) < 12:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points, None)
    if moved is None or status is None:
        return None
    valid = status.reshape(-1).astype(bool)
    src, dst = points.reshape(-1, 2)[valid], moved.reshape(-1, 2)[valid]
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src, dst = src[finite], dst[finite]
    if len(src) < 12:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if matrix is None or inliers is None:
        return None
    predicted = cv2.transform(src[None, :, :], matrix)[0]
    mask = inliers.reshape(-1).astype(bool)
    residual = np.linalg.norm(predicted[mask] - dst[mask], axis=1)
    scale_rotation = matrix[:2, :2]
    angle = np.degrees(np.arctan2(scale_rotation[1, 0], scale_rotation[0, 0]))
    return {
        "translation_px": float(np.linalg.norm(matrix[:, 2])),
        "rotation_deg": float(abs(angle)),
        "inlier_ratio": float(mask.mean()),
        "residual_px": float(np.median(residual)) if len(residual) else float("inf"),
    }


def classify_camera_motion(measurements: list[dict], image_diagonal: float) -> tuple[str, bool, str, str]:
    """综合分位数和运动段比例分类相机，识别间歇性左右摇摄。"""
    if not measurements:
        return "unknown", False, "disabled", "光流有效样本不足，默认关闭米制动态标定。"
    trans = float(np.median([m["translation_px"] for m in measurements]))
    rot = float(np.median([m["rotation_deg"] for m in measurements]))
    inlier = float(np.median([m["inlier_ratio"] for m in measurements]))
    residual = float(np.median([m["residual_px"] for m in measurements]))
    translations = np.asarray([m["translation_px"] for m in measurements], np.float32)
    p75_normalized = float(np.percentile(translations, 75)) / max(image_diagonal, 1.0)
    moving_ratio = float(np.mean(translations / max(image_diagonal, 1.0) >= .0015))
    normalized = trans / max(image_diagonal, 1.0)
    if normalized < .0015 and p75_normalized < .0025 and moving_ratio < .20 and rot < .08 and residual < 1.8:
        return "fixed", True, "static", "相机近似固定：一个经验证的静态 H 可覆盖全片。"
    stable_global_motion = float(np.mean([
        m["inlier_ratio"] >= .50 and m["residual_px"] <= 3.0 for m in measurements
    ])) >= .60
    if stable_global_motion and (moving_ratio >= .20 or p75_normalized >= .0025):
        return "pan_rotate", True, "dynamic_keyframes", "检测到间歇性云台摇摄/左右转向：以关键帧 H 动态更新，禁止沿用单一首帧 H。"
    return "handheld_translate", False, "manual_keyframes", "运动不满足稳定全局模型；仅建议密集人工关键帧标定，否则关闭米制结果。"


def inspect_video_health(video_path: str | Path, sample_pairs: int = 72) -> MotionHealth:
    """执行 Stage A：跨全片抽样并判断相机运动与标定策略。"""
    meta = read_video_metadata(video_path)
    cap = cv2.VideoCapture(str(video_path))
    measurements: list[dict] = []
    step = max(1, int(round(meta.fps * 0.25)))
    for index in uniform_frame_indices(meta.frame_count - step, sample_pairs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok1, first = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, index + step)
        ok2, second = cap.read()
        if ok1 and ok2:
            result = estimate_pair_motion(first, second)
            if result:
                measurements.append(result)
    cap.release()
    motion, usable, mode, note = classify_camera_motion(measurements, float(np.hypot(meta.width, meta.height)))
    median = lambda key, default=0.0: float(np.median([m[key] for m in measurements])) if measurements else default
    translations = [m["translation_px"] for m in measurements]
    diagonal = float(np.hypot(meta.width, meta.height))
    return MotionHealth(
        meta, motion, usable, mode, median("translation_px"), median("rotation_deg"),
        median("inlier_ratio"), median("residual_px", 0.0), len(measurements), note,
        float(np.percentile(translations, 75)) if translations else 0.0,
        float(np.percentile(translations, 90)) if translations else 0.0,
        float(np.mean(np.asarray(translations) / max(diagonal, 1.0) >= .0015)) if translations else 0.0,
    )
