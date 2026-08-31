from __future__ import annotations

"""Bridge to the canonical tracking helpers without copying their implementation."""

import sys
from pathlib import Path

TRACKING_ROOT = Path(__file__).resolve().parents[2] / "tracking"
if str(TRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKING_ROOT))

from tracking_lib.actor import BoxObservation, attribute_actor, build_global_boxes, interpolate_ball  # noqa: E402,F401
from tracking_lib.homography import *  # noqa: E402,F401,F403
from tracking_lib.team_features import (  # noqa: E402,F401
    aggregate_features, jersey_feature, jersey_pixel_mask, representative_hsv,
    suggested_color_name, torso_feature,
)
