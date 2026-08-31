import numpy as np
import cv2

from onboard.team_colors import kmeans_labels, silhouette_score
from tracking_lib.team_features import representative_hsv, suggested_color_name


def test_two_separated_color_groups_prefer_high_silhouette():
    rng = np.random.default_rng(4)
    features = np.vstack([rng.normal(0, .02, (20, 4)), rng.normal(1, .02, (20, 4))]).astype("float32")
    labels, _ = kmeans_labels(features, 2)
    assert silhouette_score(features, labels) > .9


def test_yellow_jersey_is_not_removed_as_turf_or_reported_red():
    hsv = np.full((24, 16, 3), (30, 220, 230), dtype=np.uint8)
    crop = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert suggested_color_name(representative_hsv([crop])) == "yellow"
