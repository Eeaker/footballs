from __future__ import annotations

from engine.match_analysis.analysis_lib.semantic_events import derive_semantic_events


def test_derives_shield_counterpress_and_goal_candidates() -> None:
    fps = 10.0
    team_map = {1: "team_a", 2: "team_b", 3: "team_a"}
    positions = {}
    for frame in range(0, 31):
        positions[(frame, 1)] = (10.0, 10.0)
        positions[(frame, 2)] = (11.0, 10.0)
        positions[(frame, 3)] = (12.0, 10.0)
    possessions = [
        {"global_id": "1", "team_id": "team_a", "start_frame_proc": "0", "end_frame_proc": "9"},
        {"global_id": "2", "team_id": "team_b", "start_frame_proc": "10", "end_frame_proc": "19"},
        {"global_id": "3", "team_id": "team_a", "start_frame_proc": "20", "end_frame_proc": "30"},
    ]
    transitions = [
        {"from_global_id": "1", "to_global_id": "2", "from_team_id": "team_a", "to_team_id": "team_b", "release_frame_proc": "9", "receive_frame_proc": "10", "classification": "opponent_possession_change", "start_x_m": "10", "start_y_m": "10"},
        {"from_global_id": "2", "to_global_id": "3", "from_team_id": "team_b", "to_team_id": "team_a", "release_frame_proc": "19", "receive_frame_proc": "20", "classification": "opponent_possession_change", "start_x_m": "11", "start_y_m": "10"},
    ]
    stage4 = [{"event_type": "射门_大力踢球", "event_frame_proc": 25, "event_id": 1, "score": 2.0}]
    ball = {26: (44.4, 12.5), 27: (44.9, 12.5)}

    rows = derive_semantic_events(
        fps=fps,
        duration_seconds=4.0,
        field_length_m=45.0,
        field_width_m=25.0,
        team_map=team_map,
        positions=positions,
        possessions=possessions,
        transitions=transitions,
        stage4_events=stage4,
        ball_metric_by_frame=ball,
    )
    types = {row["event_type"] for row in rows}
    assert "shielding_under_pressure" in types
    assert "counterpress_recovery" in types
    assert "goal_candidate" in types
    recovery = next(row for row in rows if row["event_type"] == "counterpress_recovery")
    assert recovery["primary_global_id"] == 3
    assert recovery["window_seconds"] == 5.0

