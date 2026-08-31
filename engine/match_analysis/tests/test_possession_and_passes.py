from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis_lib.io import BallPoint, MOTBox
from analysis_lib.passes import (
    build_statistics,
    detect_active_passes,
    detect_pass_review_candidates,
    detect_possession_transitions,
)
from analysis_lib.possession import match_ball_to_players, stable_possessions


class IdentityProvider:
    def at_processed_frame(self, frame_proc: int, vid_stride: int):
        return np.eye(3)


def box(frame: int, identity: int, foot_x: float, foot_y: float) -> MOTBox:
    return MOTBox(frame, identity, foot_x - 1, foot_y - 2, 2, 2, .9)


def test_three_consecutive_frames_and_same_team_switch_is_successful_pass():
    players = {}
    balls = {}
    for frame in range(3):
        players[frame] = [box(frame, 1, 0, 0), box(frame, 2, 10, 0)]
        balls[frame] = BallPoint(frame, .5, 0, "observed")
    for frame in range(3, 6):
        players[frame] = [box(frame, 1, 0, 0), box(frame, 2, 10, 0)]
        balls[frame] = BallPoint(frame, 10.5, 0, "observed")
    matches, _ = match_ball_to_players(
        players, balls, IdentityProvider(), {1: "team_0", 2: "team_0"},
        max_distance_m=1.5, vid_stride=1,
    )
    intervals, stable = stable_possessions(matches, 3)
    transitions = detect_possession_transitions(intervals, 5)
    events = detect_active_passes(transitions, min_displacement_m=0.5, fps=10.0)
    assert [row.global_id for row in intervals] == [1, 2]
    assert len(stable) == 6
    assert len(events) == 1
    assert len(transitions) == 1
    assert events[0].classification == "active_directed_pass_candidate"
    summary, matrix, _ = build_statistics(events)
    assert summary[0]["active_directed_passes"] == 1
    assert matrix[0]["active_directed_passes"] == 1


def test_all_a_to_b_changes_remain_transitions_but_jitter_is_not_a_pass():
    intervals = []
    players = {}
    balls = {}
    for frame in range(3):
        players[frame] = [box(frame, 1, 0, 0), box(frame, 2, .1, 0)]
        balls[frame] = BallPoint(frame, 0, 0, "observed")
    for frame in range(3, 6):
        players[frame] = [box(frame, 1, 0, 0), box(frame, 2, .1, 0)]
        balls[frame] = BallPoint(frame, .1, 0, "observed")
    matches, _ = match_ball_to_players(
        players, balls, IdentityProvider(), {1: "team_0", 2: "team_0"},
        max_distance_m=1.5, vid_stride=1,
    )
    intervals, _ = stable_possessions(matches, 3)
    transitions = detect_possession_transitions(intervals, 5)
    passes = detect_active_passes(transitions, min_displacement_m=.5, fps=10.0)
    assert len(transitions) == 1
    assert transitions[0].classification == "same_team_short_or_stationary_transition"
    assert passes == []


def test_gray_zone_is_exportable_but_never_a_formal_pass():
    players = {}
    balls = {}
    for frame in range(3):
        players[frame] = [box(frame, 1, 0, 0), box(frame, 2, .3, 0)]
        balls[frame] = BallPoint(frame, 0, 0, "observed")
    for frame in range(3, 6):
        players[frame] = [box(frame, 1, 0, 0), box(frame, 2, .3, 0)]
        balls[frame] = BallPoint(frame, .3, 0, "observed")
    matches, _ = match_ball_to_players(
        players, balls, IdentityProvider(), {1: "team_0", 2: "team_0"},
        max_distance_m=1.5, vid_stride=1,
    )
    intervals, _ = stable_possessions(matches, 3)
    transitions = detect_possession_transitions(intervals, 5)
    formal = detect_active_passes(transitions, min_displacement_m=.5, fps=10.0)
    review = detect_pass_review_candidates(
        transitions, review_min_displacement_m=.25, formal_min_displacement_m=.5,
    )
    assert formal == []
    assert len(review) == 1
    assert review[0]["review_classification"] == "directed_pass_gray_zone"


def test_opponent_transition_never_enters_pass_network_even_with_displacement():
    players = {}
    balls = {}
    for frame in range(3):
        players[frame] = [box(frame, 1, 0, 0), box(frame, 2, 10, 0)]
        balls[frame] = BallPoint(frame, 0, 0, "observed")
    for frame in range(3, 6):
        players[frame] = [box(frame, 1, 0, 0), box(frame, 2, 10, 0)]
        balls[frame] = BallPoint(frame, 10, 0, "observed")
    matches, _ = match_ball_to_players(
        players, balls, IdentityProvider(), {1: "team_0", 2: "team_1"},
        max_distance_m=1.5, vid_stride=1,
    )
    intervals, _ = stable_possessions(matches, 3)
    transitions = detect_possession_transitions(intervals, 5)
    passes = detect_active_passes(transitions, min_displacement_m=.5, fps=10.0)
    assert transitions[0].classification == "opponent_possession_change"
    assert passes == []


def test_two_frames_do_not_create_possession():
    players = {frame: [box(frame, 1, 0, 0)] for frame in range(2)}
    balls = {frame: BallPoint(frame, .5, 0, "observed") for frame in range(2)}
    matches, _ = match_ball_to_players(
        players, balls, IdentityProvider(), {1: "team_0"}, max_distance_m=1.5, vid_stride=1,
    )
    intervals, _ = stable_possessions(matches, 3)
    assert intervals == []


def test_exactly_1_5_m_is_excluded_by_strict_contract():
    players = {0: [box(0, 1, 0, 0)]}
    balls = {0: BallPoint(0, 1.5, 0, "observed")}
    matches, _ = match_ball_to_players(
        players, balls, IdentityProvider(), {1: "team_0"}, max_distance_m=1.5, vid_stride=1,
    )
    assert matches == []
