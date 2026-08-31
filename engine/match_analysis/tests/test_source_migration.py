from __future__ import annotations

from pathlib import Path

from analysis_lib import tracking_adapter

ROOT = Path(__file__).resolve().parents[1]

def test_tracking_adapter_points_to_canonical_tracking_tree():
    assert tracking_adapter.TRACKING_ROOT.name == "tracking"
    assert (tracking_adapter.TRACKING_ROOT / "tracking_lib" / "actor.py").is_file()
    assert (tracking_adapter.TRACKING_ROOT / "tracking_lib" / "homography.py").is_file()
    assert (tracking_adapter.TRACKING_ROOT / "tracking_lib" / "team_features.py").is_file()

def test_copied_tracking_helpers_are_not_bundled_in_analysis():
    assert not (ROOT / "analysis_lib" / "migrated").exists()

def test_vendor_pass_rules_keep_separate_boundary():
    assert (ROOT / "analysis_lib" / "vendor" / "tryolabs_pass_rules.py").is_file()
    assert (ROOT / "licenses" / "TRYOLABS_MIT.txt").is_file()
