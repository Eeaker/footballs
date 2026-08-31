from export_task3_final_demo import (
    build_ball_visual_track,
    build_possession_index,
    resolve_ball_state,
)


def test_short_internal_ball_gap_is_visually_interpolated_without_claiming_observation():
    observed = {10: (10.0, 20.0), 13: (16.0, 26.0)}
    track = build_ball_visual_track(observed, max_interpolation_gap_frames=3)

    assert track[10] == (10.0, 20.0, "observed")
    assert track[11] == (12.0, 22.0, "visual_interp")
    assert track[12] == (14.0, 24.0, "visual_interp")
    assert track[13] == (16.0, 26.0, "observed")


def test_long_gap_is_not_fabricated_and_degrades_to_last_seen_then_unobserved():
    track = build_ball_visual_track({10: (10.0, 20.0), 30: (30.0, 40.0)}, 3)

    assert 11 not in track
    assert resolve_ball_state(12, track, last_seen=(10, 10.0, 20.0), hold_frames=3)[2] == "last_seen"
    assert resolve_ball_state(14, track, last_seen=(10, 10.0, 20.0), hold_frames=3)[2] == "unobserved"


def test_possession_index_fills_confirmed_interval_and_keeps_gaps_unconfirmed():
    intervals = [{
        "global_id": "15", "team_id": "team_1",
        "start_frame_proc": "100", "confirmed_frame_proc": "102", "end_frame_proc": "106",
    }]
    index = build_possession_index(intervals)

    assert 100 not in index
    assert index[102] == (15, "team_1")
    assert index[106] == (15, "team_1")
    assert 107 not in index
