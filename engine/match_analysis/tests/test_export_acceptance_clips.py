from __future__ import annotations

from export_acceptance_clips import frame_bounds, select_events


def test_select_events_preserves_deterministic_sample_order():
    events = [{"pass_id": str(index)} for index in range(5)]
    sample = [{"pass_id": "3"}, {"pass_id": "1"}]
    assert [row["pass_id"] for row in select_events(events, sample)] == ["3", "1"]


def test_clip_bounds_cover_release_and_receive():
    assert frame_bounds(100, 120, 1000, 10.0, 1.0, 2.0) == (90, 140)
