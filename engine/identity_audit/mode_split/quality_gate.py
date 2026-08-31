from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

TRACKING_ROOT = Path(__file__).resolve().parents[3] / "tracking"
if str(TRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKING_ROOT))

from tracking_lib.team_features import jersey_feature, jersey_pixel_mask


@dataclass(frozen=True)
class TorsoQuality:
    accepted: bool
    reason: str
    torso_occlusion: float
    visible_fraction: float
    informative_fraction: float
    sharpness: float


def torso_rect(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Central upper-torso ROI, matching the canonical tracking jersey extractor."""
    x, y, w, h = box
    return x + .24 * w, y + .10 * h, x + .76 * w, y + .58 * h


def intersection_over_target(
    target: tuple[float, float, float, float],
    occluder: tuple[float, float, float, float],
) -> float:
    tx1, ty1, tx2, ty2 = target
    ox, oy, ow, oh = occluder
    ox1, oy1, ox2, oy2 = ox, oy, ox + ow, oy + oh
    iw = max(0.0, min(tx2, ox2) - max(tx1, ox1))
    ih = max(0.0, min(ty2, oy2) - max(ty1, oy1))
    area = max((tx2 - tx1) * (ty2 - ty1), 1e-9)
    return iw * ih / area


def quality_gated_torso_feature(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    other_boxes: list[tuple[float, float, float, float]],
    *,
    maximum_torso_occlusion: float = .18,
    minimum_visible_fraction: float = .92,
    minimum_informative_fraction: float = .18,
    minimum_sharpness: float = 8.0,
) -> tuple[np.ndarray | None, TorsoQuality]:
    """Extract a jersey feature only when the upper torso is usable.

    No team colour or colour name is supplied.  Frames rejected here remain in
    the MOT; they simply become unknown for team-switch reasoning.
    """
    height, width = frame.shape[:2]
    raw = torso_rect(box)
    x1, y1, x2, y2 = raw
    raw_area = max((x2 - x1) * (y2 - y1), 1e-9)
    clipped = (
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )
    cx1, cy1, cx2, cy2 = clipped
    visible = max(0, cx2 - cx1) * max(0, cy2 - cy1) / raw_area
    overlap = max((intersection_over_target(raw, item) for item in other_boxes), default=0.0)
    if cx2 - cx1 < 8 or cy2 - cy1 < 12:
        return None, TorsoQuality(False, "too_small", overlap, visible, 0.0, 0.0)
    if visible < minimum_visible_fraction:
        return None, TorsoQuality(False, "frame_edge", overlap, visible, 0.0, 0.0)
    if overlap > maximum_torso_occlusion:
        return None, TorsoQuality(False, "person_overlap", overlap, visible, 0.0, 0.0)

    crop = frame[cy1:cy2, cx1:cx2]
    scale = min(1.0, 32.0 / max(crop.shape[:2]))
    if scale < 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    informative = float(jersey_pixel_mask(hsv).mean())
    sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
    if informative < minimum_informative_fraction:
        return None, TorsoQuality(False, "low_information", overlap, visible, informative, sharpness)
    if sharpness < minimum_sharpness:
        return None, TorsoQuality(False, "blurred", overlap, visible, informative, sharpness)
    return jersey_feature(crop), TorsoQuality(True, "accepted", overlap, visible, informative, sharpness)
