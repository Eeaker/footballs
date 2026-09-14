from __future__ import annotations

import cv2
import numpy as np


def jersey_pixel_mask(hsv: np.ndarray) -> np.ndarray:
    """保留球衣有效像素，同时排除草地和几乎无信息的暗像素。"""
    hue, saturation, value = cv2.split(hsv)
    # 黄色球衣与草地在 HSV 中相邻，不能从 28 开始整段删除；否则黄色会被
    # 当成背景，只剩皮肤/阴影，最终色名很容易被误报为红色。
    turf = (hue >= 38) & (hue <= 100) & (saturation >= 45)
    return (~turf) & (value >= 35)


def jersey_feature(crop: np.ndarray) -> np.ndarray:
    """提取兼容彩色与白色球衣的背景抑制 HSV 特征。"""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = jersey_pixel_mask(hsv)
    if int(mask.sum()) < 12:
        mask = hsv[..., 2] >= 25
    mask_u8 = mask.astype(np.uint8) * 255
    hist = cv2.calcHist([hsv], [0, 1], mask_u8, [24, 8], [0, 180, 0, 256]).reshape(-1)
    hist = np.sqrt(hist / max(float(hist.sum()), 1e-9))
    pixels = hsv[mask].astype(np.float32)
    saturation, value = pixels[:, 1], pixels[:, 2]
    colorful = saturation >= 45
    if colorful.any():
        angles = pixels[colorful, 0] * (2.0 * np.pi / 180.0)
        weights = saturation[colorful] / 255.0
        hue_cos = float(np.average(np.cos(angles), weights=weights))
        hue_sin = float(np.average(np.sin(angles), weights=weights))
    else:
        hue_cos = hue_sin = 0.0
    white_fraction = float(((saturation < 45) & (value >= 100)).mean())
    scalars = np.asarray([
        1.5 * hue_cos, 1.5 * hue_sin,
        float(np.median(saturation)) / 127.5,
        float(np.median(value)) / 255.0,
        2.0 * white_fraction, 2.0 * float(colorful.mean()),
    ], np.float32)
    feature = np.concatenate([hist.astype(np.float32), scalars])
    return feature / max(float(np.linalg.norm(feature)), 1e-9)


def torso_feature(frame: np.ndarray, xyxy: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray] | None:
    """从人员框中央上身提取球衣核心，减少手臂、短裤和背景干扰。"""
    x1, y1, x2, y2 = map(int, xyxy)
    height, width = frame.shape[:2]
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    if x2 - x1 < 8 or y2 - y1 < 16:
        return None
    box_h, box_w = y2 - y1, x2 - x1
    torso = frame[y1 + int(.10 * box_h):y1 + int(.58 * box_h),
                  x1 + int(.24 * box_w):x2 - int(.24 * box_w)]
    if torso.size == 0 or min(torso.shape[:2]) < 3:
        return None
    return jersey_feature(torso), torso.copy()


def aggregate_features(features: list[np.ndarray]) -> np.ndarray:
    """以逐维中位数聚合一条轨迹的多帧球衣特征，每条轨迹只贡献一个原型。"""
    if not features:
        return np.zeros(198, np.float32)
    result = np.median(np.asarray(features, np.float32), axis=0)
    return result.astype(np.float32) / max(float(np.linalg.norm(result)), 1e-9)


def representative_hsv(crops: list[np.ndarray]) -> list[float]:
    """从球衣有效像素估计代表色，避免草地和阴影主导色块。"""
    groups = []
    for crop in crops:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels = hsv[jersey_pixel_mask(hsv)]
        if len(pixels):
            groups.append(pixels)
    if not groups:
        return [0.0, 0.0, 0.0]
    pixels = np.concatenate(groups).astype(np.float32)
    saturation, value = pixels[:, 1], pixels[:, 2]
    white_fraction = float(((saturation < 45) & (value >= 100)).mean())
    if white_fraction >= .55:
        neutral = pixels[(saturation < 55) & (value >= 80)]
        return [0.0, float(np.median(neutral[:, 1])), float(np.median(neutral[:, 2]))]
    colorful = pixels[saturation >= 45]
    if not len(colorful):
        return np.median(pixels, axis=0).tolist()
    angles = colorful[:, 0] * (2.0 * np.pi / 180.0)
    weights = colorful[:, 1] / 255.0
    angle = float(np.arctan2(np.sum(np.sin(angles) * weights), np.sum(np.cos(angles) * weights)))
    hue = (angle % (2.0 * np.pi)) * 180.0 / (2.0 * np.pi)
    return [hue, float(np.median(colorful[:, 1])), float(np.median(colorful[:, 2]))]


def suggested_color_name(hsv: list[float]) -> str:
    """把代表 HSV 转成仅供人工参考的通用颜色名。"""
    hue, saturation, value = hsv
    if value < 55: return "black"
    if saturation < 45: return "white" if value >= 145 else "gray"
    if hue < 10 or hue >= 170: return "red"
    if hue < 24: return "orange"
    if hue < 42: return "yellow"
    if hue < 88: return "green"
    if hue < 135: return "blue"
    if hue < 165: return "purple"
    return "red"
