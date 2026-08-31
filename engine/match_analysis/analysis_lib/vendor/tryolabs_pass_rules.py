"""Directly ported rules from Tryolabs soccer/pass_event.py (MIT, commit b00c7c7).

The upstream ``PassEvent.validate_pass`` rejects same-player transitions and
requires equal teams.  match analysis calls these rules after its metric, three-frame
possession detector.  Turnovers are retained separately because the local task
also asks for a pass-success denominator.
"""

from __future__ import annotations


def is_player_transition(start_player_id: int, end_player_id: int) -> bool:
    """Port of upstream's first validation guard: same player is not a pass."""
    return start_player_id != end_player_id


def validate_successful_pass(
    start_player_id: int, end_player_id: int, start_team: str, end_team: str,
) -> bool:
    """Port of Tryolabs PassEvent.validate_pass using serializable IDs/teams."""
    if start_player_id == end_player_id:
        return False
    if start_team != end_team:
        return False
    return True
