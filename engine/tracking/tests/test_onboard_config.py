from onboard.config_builder import build_pipeline_config, safe_venue_name
from onboard.models import CalibrationResult, MotionHealth, VideoMetadata
from tracking_lib.config import argparse_defaults


def test_config_maps_to_pipeline_arguments():
    meta = VideoMetadata("x.mp4", 30, 1920, 1080, 300, 10)
    health = MotionHealth(meta, "fixed", True, "static", 0, 0, 1, 0, 10, "ok")
    config = build_pipeline_config("A/B", "x.mp4", health, None, CalibrationResult(False, "disabled"),
                                   "tracker.yaml", None, 16, False)
    defaults = argparse_defaults(config)
    assert safe_venue_name("A/B") == "A_B"
    assert defaults["expected_on_field_players"] == 16
    assert defaults["team_clusters"] == 2
    assert defaults["pre_sec"] == defaults["post_sec"] == 15
    assert defaults["tracker_config"].endswith("tracker.yaml")

