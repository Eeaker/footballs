from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math

from .vendor.tryolabs_pass_rules import is_player_transition
from .possession import PossessionInterval


@dataclass(frozen=True)
class PossessionTransition:
    transition_id: int
    from_global_id: int
    to_global_id: int
    from_team_id: str
    to_team_id: str
    release_frame_proc: int
    receive_frame_proc: int
    receive_confirmed_frame_proc: int
    transfer_gap_frames: int
    source_evidence_frames: int
    receiver_evidence_frames: int
    start_x_m: float
    start_y_m: float
    end_x_m: float
    end_y_m: float
    dx_m: float
    dy_m: float
    displacement_m: float
    classification: str


@dataclass(frozen=True)
class PassEvent:
    pass_id: int
    transition_id: int
    from_global_id: int
    to_global_id: int
    team_id: str
    release_frame_proc: int
    receive_frame_proc: int
    receive_confirmed_frame_proc: int
    transfer_gap_frames: int
    start_x_m: float
    start_y_m: float
    end_x_m: float
    end_y_m: float
    dx_m: float
    dy_m: float
    distance_m: float
    direction_angle_deg: float
    transfer_speed_mps: float
    classification: str
    intent_proxy: str


def detect_possession_transitions(
    intervals: list[PossessionInterval], max_transfer_gap_frames: int,
    *, short_displacement_m: float = .5,
) -> list[PossessionTransition]:
    """Keep every stable A→B change; do not call all changes passes."""
    result: list[PossessionTransition] = []
    for previous, current in zip(intervals, intervals[1:]):
        if not is_player_transition(previous.global_id, current.global_id):
            continue
        gap = current.start_frame_proc - previous.end_frame_proc - 1
        if gap < 0 or gap > max_transfer_gap_frames:
            continue
        dx = current.start_ball_x_m - previous.end_ball_x_m
        dy = current.start_ball_y_m - previous.end_ball_y_m
        displacement = math.hypot(dx, dy)
        if "unassigned" in {previous.team_id, current.team_id}:
            classification = "unknown_team_possession_change"
        elif previous.team_id != current.team_id:
            classification = "opponent_possession_change"
        elif displacement < short_displacement_m:
            classification = "same_team_short_or_stationary_transition"
        else:
            classification = "same_team_directed_transition"
        result.append(PossessionTransition(
            len(result), previous.global_id, current.global_id,
            previous.team_id, current.team_id,
            previous.end_frame_proc, current.start_frame_proc, current.confirmed_frame_proc, gap,
            previous.evidence_frames, current.evidence_frames,
            previous.end_ball_x_m, previous.end_ball_y_m,
            current.start_ball_x_m, current.start_ball_y_m,
            dx, dy, displacement, classification,
        ))
    return result


def detect_active_passes(
    transitions: list[PossessionTransition], *, min_displacement_m: float, fps: float,
) -> list[PassEvent]:
    """Return same-team, stable, directed transfers as auditable pass candidates.

    Tactical intent is not directly observable from single-camera tracks. The
    explicit proxy is stable possession at both ends, same team, and a minimum
    metric displacement. Human review remains authoritative.
    """
    if min_displacement_m <= 0:
        raise ValueError("min_displacement_m must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    result: list[PassEvent] = []
    for transition in transitions:
        if transition.from_team_id == "unassigned":
            continue
        if transition.from_team_id != transition.to_team_id:
            continue
        if transition.displacement_m < min_displacement_m:
            continue
        elapsed_frames = max(1, transition.receive_frame_proc - transition.release_frame_proc)
        result.append(PassEvent(
            len(result), transition.transition_id,
            transition.from_global_id, transition.to_global_id, transition.from_team_id,
            transition.release_frame_proc, transition.receive_frame_proc,
            transition.receive_confirmed_frame_proc, transition.transfer_gap_frames,
            transition.start_x_m, transition.start_y_m, transition.end_x_m, transition.end_y_m,
            transition.dx_m, transition.dy_m, transition.displacement_m,
            math.degrees(math.atan2(transition.dy_m, transition.dx_m)),
            transition.displacement_m / (elapsed_frames / fps),
            "active_directed_pass_candidate",
            "stable_A_and_B+same_team+metric_displacement;human_review_required",
        ))
    return result


def detect_pass_review_candidates(
    transitions: list[PossessionTransition], *, review_min_displacement_m: float,
    formal_min_displacement_m: float,
) -> list[dict]:
    """Return a non-scoring gray zone for review; never feed it into the network."""
    if not 0 < review_min_displacement_m < formal_min_displacement_m:
        raise ValueError("review threshold must be positive and below the formal threshold")
    rows = []
    for transition in transitions:
        if transition.from_team_id in {"unassigned", ""}:
            continue
        if transition.from_team_id != transition.to_team_id:
            continue
        if not review_min_displacement_m <= transition.displacement_m < formal_min_displacement_m:
            continue
        rows.append({
            **transition.__dict__,
            "review_classification": "directed_pass_gray_zone",
            "review_reason": (
                f"same_team_stable_transfer; displacement in "
                f"[{review_min_displacement_m},{formal_min_displacement_m})m; excluded_from_network"
            ),
        })
    return rows


def build_statistics(events: list[PassEvent]) -> tuple[list[dict], list[dict], dict]:
    counts = Counter()
    distances = Counter()
    matrix = Counter()
    players_by_team: dict[str, set[int]] = defaultdict(set)
    for event in events:
        counts[event.team_id] += 1
        distances[event.team_id] += event.distance_m
        matrix[(event.team_id, event.from_global_id, event.to_global_id)] += 1
        players_by_team[event.team_id].update((event.from_global_id, event.to_global_id))
    summary = [{
        "team_id": team,
        "active_directed_passes": counts[team],
        "total_pass_distance_m": round(distances[team], 3),
        "mean_pass_distance_m": round(distances[team] / counts[team], 3),
    } for team in sorted(counts)]
    matrix_long = [{
        "team_id": team, "from_global_id": source, "to_global_id": target,
        "active_directed_passes": count,
    } for (team, source, target), count in sorted(matrix.items())]
    matrix_json = {}
    for team, players in sorted(players_by_team.items()):
        ids = sorted(players)
        matrix_json[team] = {
            "global_ids": ids,
            "matrix": [[matrix[(team, source, target)] for target in ids] for source in ids],
        }
    return summary, matrix_long, matrix_json
