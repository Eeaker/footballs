from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _overlap_ratio(box: tuple[float, float, float, float], others: Iterable[tuple[float, float, float, float]]) -> float:
    x, y, w, h = box
    best = 0.0
    for xx, yy, ww, hh in others:
        inter = max(0.0, min(x + w, xx + ww) - max(x, xx)) * max(0.0, min(y + h, yy + hh) - max(y, yy))
        best = max(best, inter / max(w * h, 1e-9))
    return best


def _mot_candidates(path: Path, gids: set[int], limit: int = 80) -> list[tuple[int, int, float, float, float, float, float]]:
    by_gid: dict[int, list[tuple[int, int, float, float, float, float, float]]] = {gid: [] for gid in gids}
    with path.open(encoding="utf-8-sig") as handle:
        for values in csv.reader(handle):
            if len(values) < 6:
                continue
            frame, gid = int(float(values[0])), int(float(values[1]))
            if gid not in gids:
                continue
            confidence = float(values[6]) if len(values) > 6 else 1.0
            by_gid[gid].append((frame, gid, *map(float, values[2:6]), confidence))
    selected = []
    per_gid = max(4, limit // max(1, len(gids)))
    for rows in by_gid.values():
        if not rows:
            continue
        indices = np.unique(np.linspace(0, len(rows) - 1, min(per_gid, len(rows))).astype(int))
        selected.extend(rows[index] for index in indices)
    return selected[:limit]


def select_clear_avatar(video: Path, mot: Path, global_ids: Iterable[int]) -> bytes:
    """Choose a real, sharp, unobstructed trajectory frame without persisting a local avatar."""
    gids = {int(value) for value in global_ids}
    rows = _mot_candidates(mot, gids)
    if not rows:
        raise FileNotFoundError("没有可用球员轨迹帧")
    by_frame: dict[int, list[tuple]] = {}
    with mot.open(encoding="utf-8-sig") as handle:
        for values in csv.reader(handle):
            if len(values) >= 6:
                frame = int(float(values[0]))
                if any(row[0] == frame for row in rows):
                    by_frame.setdefault(frame, []).append(tuple(map(float, values[2:6])))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("无法打开比赛视频")
    best: tuple[float, np.ndarray] | None = None
    try:
        for frame, _gid, x, y, w, h, confidence in rows:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame - 1))
            ok, image = cap.read()
            if not ok:
                continue
            ih, iw = image.shape[:2]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(iw, int(x + w)), min(ih, int(y + h))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0 or min(crop.shape[:2]) < 24:
                continue
            others = [box for box in by_frame.get(frame, []) if box != (x, y, w, h)]
            overlap = _overlap_ratio((x, y, w, h), others)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            exposure = 1.0 - min(1.0, abs(float(gray.mean()) - 128.0) / 128.0)
            score = np.log1p(sharpness) * confidence * (1.0 - min(overlap, 0.95)) * exposure * min(1.0, h / 160.0)
            if best is None or score > best[0]:
                best = (float(score), crop.copy())
    finally:
        cap.release()
    if best is None:
        raise FileNotFoundError("没有达到清晰度要求的球员帧")
    ok, jpeg = cv2.imencode(".jpg", best[1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("球员头像编码失败")
    return jpeg.tobytes()

