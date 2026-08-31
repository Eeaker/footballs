from __future__ import annotations

import numpy as np


class HomographyProvider:
    """为静态或动态关键帧标定按帧提供单应矩阵。"""

    def __init__(self, calibration: dict):
        self.mode = calibration.get("mode", "disabled")
        self.keyframes = sorted(calibration.get("keyframes", []), key=lambda row: row["frame_index"])

    def at(self, raw_frame: int) -> np.ndarray | None:
        """在相邻人工验证锚点之间自动插值 H。"""
        if not self.keyframes:
            return None
        if self.mode == "static" or len(self.keyframes) == 1:
            return np.asarray(self.keyframes[0]["homography"], dtype=np.float64)
        if raw_frame <= self.keyframes[0]["frame_index"]:
            return np.asarray(self.keyframes[0]["homography"], dtype=np.float64)
        if raw_frame >= self.keyframes[-1]["frame_index"]:
            return np.asarray(self.keyframes[-1]["homography"], dtype=np.float64)
        for left, right in zip(self.keyframes, self.keyframes[1:]):
            if left["frame_index"] <= raw_frame <= right["frame_index"]:
                first = np.asarray(left["homography"], dtype=np.float64)
                second = np.asarray(right["homography"], dtype=np.float64)
                first /= first[2, 2]; second /= second[2, 2]
                ratio = (raw_frame - left["frame_index"]) / (right["frame_index"] - left["frame_index"])
                result = first * (1 - ratio) + second * ratio
                return result / result[2, 2]
        return None


def project_footpoint(homography: np.ndarray, x: float, y: float, width: float, height: float) -> tuple[float, float]:
    """将人员框底部中心映射到球场米制平面。"""
    point = np.asarray([x + width / 2, y + height, 1.0], dtype=np.float64)
    mapped = homography @ point
    if abs(mapped[2]) < 1e-9:
        return float("nan"), float("nan")
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])
