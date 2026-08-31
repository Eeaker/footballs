from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import HomographyProvider, project_point
from .io import BallPoint, MOTBox


@dataclass(frozen=True)
class FrameMatch:
    frame_proc: int
    global_id: int
    team_id: str
    distance_m: float
    ball_x_m: float
    ball_y_m: float
    player_x_m: float
    player_y_m: float
    ball_source: str


@dataclass(frozen=True)
class PossessionInterval:
    possession_id: int
    global_id: int
    team_id: str
    start_frame_proc: int
    confirmed_frame_proc: int
    end_frame_proc: int
    evidence_frames: int
    min_distance_m: float
    max_distance_m: float
    mean_distance_m: float
    start_ball_x_m: float
    start_ball_y_m: float
    end_ball_x_m: float
    end_ball_y_m: float


def match_ball_to_players(
    players_by_frame: dict[int, list[MOTBox]],
    ball_by_frame: dict[int, BallPoint],
    provider: HomographyProvider,
    team_map: dict[int, str],
    *,
    max_distance_m: float,
    vid_stride: int,
) -> tuple[list[FrameMatch], dict]:
    matches: list[FrameMatch] = []
    frames_with_matrix = frames_with_players = 0
    ball_outside_field = player_points_outside_field = 0
    for frame in sorted(ball_by_frame):
        players = players_by_frame.get(frame, [])
        if not players:
            continue
        frames_with_players += 1
        matrix = provider.at_processed_frame(frame, vid_stride)
        if matrix is None:
            continue
        frames_with_matrix += 1
        ball = ball_by_frame[frame]
        ball_m = project_point(matrix, ball.x, ball.y)
        if ball_m is None:
            continue
        if hasattr(provider, "in_field") and not provider.in_field(ball_m):
            ball_outside_field += 1
            continue
        candidates = []
        for player in players:
            player_m = project_point(matrix, *player.footpoint_px)
            if player_m is None:
                continue
            if hasattr(provider, "in_field") and not provider.in_field(player_m):
                player_points_outside_field += 1
                continue
            distance = float(np.hypot(ball_m[0] - player_m[0], ball_m[1] - player_m[1]))
            candidates.append((distance, player, player_m))
        if not candidates:
            continue
        distance, player, player_m = min(candidates, key=lambda item: (item[0], item[1].global_id))
        if distance < max_distance_m:  # task wording is strict: < 1.5 m
            matches.append(FrameMatch(
                frame, player.global_id, team_map.get(player.global_id, "unassigned"), distance,
                ball_m[0], ball_m[1], player_m[0], player_m[1], ball.source,
            ))
    report = {
        "ball_frames": len(ball_by_frame), "ball_frames_with_players": frames_with_players,
        "ball_frames_with_homography": frames_with_matrix, "candidate_match_frames": len(matches),
        "ball_points_outside_calibrated_field": ball_outside_field,
        "player_points_outside_calibrated_field": player_points_outside_field,
    }
    return matches, report


def stable_possessions(matches: list[FrameMatch], min_consecutive_frames: int) -> tuple[list[PossessionInterval], list[FrameMatch]]:
    intervals: list[PossessionInterval] = []
    stable_frames: list[FrameMatch] = []
    run: list[FrameMatch] = []

    def flush() -> None:
        if len(run) < min_consecutive_frames:
            return
        identity = run[0].global_id
        distances = [row.distance_m for row in run]
        intervals.append(PossessionInterval(
            len(intervals), identity, run[0].team_id, run[0].frame_proc,
            run[min_consecutive_frames - 1].frame_proc, run[-1].frame_proc, len(run),
            min(distances), max(distances), float(np.mean(distances)),
            run[0].ball_x_m, run[0].ball_y_m, run[-1].ball_x_m, run[-1].ball_y_m,
        ))
        stable_frames.extend(run)

    for row in sorted(matches, key=lambda item: item.frame_proc):
        if run and (row.global_id != run[-1].global_id or row.frame_proc != run[-1].frame_proc + 1):
            flush()
            run = []
        run.append(row)
    flush()
    return intervals, stable_frames
