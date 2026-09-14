from __future__ import annotations

import cv2
import numpy as np

from .homography import HomographyProvider


class FieldGeometryProvider:
    """按原始帧提供比赛区域判断，支持动态多边形或米制 Homography。"""

    def __init__(self, geometry: dict | None):
        self.geometry = geometry or {"enabled": False, "mode": "disabled"}
        self.mode = self.geometry.get("mode", "disabled")
        self.polygons = sorted(self.geometry.get("keyframes", []), key=lambda row: row["frame_index"])
        calibration = self.geometry.get("calibration", {})
        self.homography = HomographyProvider(calibration) if calibration else None

    @property
    def enabled(self) -> bool:
        return bool(self.geometry.get("enabled"))

    def polygon_at(self, raw_frame: int) -> np.ndarray | None:
        """返回图像空间可见场地多边形。

        相邻关键帧点数相同时逐点插值；因画面裁剪导致点数变化时，使用距离
        当前帧最近的完整多边形，绝不把不同语义的顶点强行一一配对。
        """
        if not self.polygons:
            return None
        if len(self.polygons) == 1 or raw_frame <= self.polygons[0]["frame_index"]:
            return np.asarray(self.polygons[0]["points"], np.float32)
        if raw_frame >= self.polygons[-1]["frame_index"]:
            return np.asarray(self.polygons[-1]["points"], np.float32)
        for left, right in zip(self.polygons, self.polygons[1:]):
            if left["frame_index"] <= raw_frame <= right["frame_index"]:
                first = np.asarray(left["points"], np.float32)
                second = np.asarray(right["points"], np.float32)
                if first.shape != second.shape:
                    return first if raw_frame - left["frame_index"] <= right["frame_index"] - raw_frame else second
                ratio = (raw_frame - left["frame_index"]) / (right["frame_index"] - left["frame_index"])
                return first * (1 - ratio) + second * ratio
        return None

    def contains(self, raw_frame: int, point: tuple[float, float]) -> bool:
        """判断人员脚点是否位于允许容差内的比赛区域。"""
        if not self.enabled:
            return True
        margin = float(self.geometry.get("margin", 0.0))
        if self.mode == "homography" and self.homography is not None:
            matrix = self.homography.at(raw_frame)
            if matrix is None:
                return False
            source = np.asarray([[point]], np.float32)
            x, y = cv2.perspectiveTransform(source, matrix)[0, 0]
            length = float(self.geometry["field_length_m"])
            width = float(self.geometry["field_width_m"])
            return -margin <= x <= length + margin and -margin <= y <= width + margin
        polygon = self.polygon_at(raw_frame)
        if polygon is None:
            return False
        return cv2.pointPolygonTest(polygon, point, True) >= -margin
