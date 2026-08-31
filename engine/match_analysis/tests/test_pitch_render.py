from __future__ import annotations

from analysis_lib.pitch_render import PitchMapper, active_event_at_frame


def test_pitch_mapper_preserves_screen_view_orientation():
    mapper = PitchMapper(100, 50, {"x_min": 0, "x_max": 45, "y_min": 0, "y_max": 25})
    assert mapper.to_px(0, 0) == (0, 0)
    assert mapper.to_px(45, 25) == (100, 50)
    assert mapper.to_px(22.5, 12.5) == (50, 25)


def test_event_is_visible_from_release_through_confirmed_receiver():
    event = {"release_frame_proc": 10, "receive_confirmed_frame_proc": 15}
    assert active_event_at_frame(event, 9) is False
    assert active_event_at_frame(event, 10) is True
    assert active_event_at_frame(event, 15) is True
    assert active_event_at_frame(event, 16) is False
