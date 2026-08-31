import pytest

from reclassify_ocr_boss_compatible import (
    apply_manual_audit, classify_voted, simultaneous_conflicts,
)


def _diagnostic(number: int, evidence: list[tuple[int, float]]) -> dict:
    return {
        "aggregated_detections": [
            {"number": number, "frame": frame, "confidence": confidence}
            for frame, confidence in evidence
        ],
    }


def test_two_strong_frames_confirm_like_frozen_boss_output():
    voted = {
        "jersey_number": 25, "winner_margin": .22, "head_confidence": .76,
        "candidates": [{"jersey_number": 25, "votes": 3}],
    }
    result = classify_voted(voted, _diagnostic(25, [(100, .72), (160, .95)]))
    assert result["status"] == "confirmed"
    assert result["support_frames"] == [100, 160]


def test_single_clear_fleeting_frame_is_tentative_not_confirmed():
    voted = {
        "jersey_number": 7, "winner_margin": .23, "head_confidence": .64,
        "candidates": [{"jersey_number": 7, "votes": 1}],
    }
    result = classify_voted(voted, _diagnostic(7, [(49955, .999)]))
    assert result["status"] == "tentative"


def test_multiple_rois_in_one_frame_still_count_as_one_vote():
    voted = {
        "jersey_number": 10, "winner_margin": .3, "head_confidence": .8,
        "candidates": [{"jersey_number": 10, "votes": 2}],
    }
    diagnostic = _diagnostic(10, [(20, .7), (20, .99)])
    result = classify_voted(voted, diagnostic)
    assert result["status"] == "tentative"
    assert result["support_frames"] == [20]


def test_close_competing_number_remains_conflict():
    voted = {
        "jersey_number": 11, "winner_margin": .03, "head_confidence": .53,
        "candidates": [
            {"jersey_number": 11, "votes": 3},
            {"jersey_number": 21, "votes": 3},
        ],
    }
    result = classify_voted(voted, _diagnostic(11, [(10, .98), (20, .9), (30, .8)]))
    assert result["status"] == "conflict"


def test_same_team_same_number_overlap_is_conflict():
    rows = [
        {"global_id": 1, "team": "blue", "predicted_number": 25, "status": "confirmed"},
        {"global_id": 2, "team": "blue", "predicted_number": 25, "status": "confirmed"},
        {"global_id": 3, "team": "yellow", "predicted_number": 25, "status": "confirmed"},
    ]
    assert simultaneous_conflicts(rows, {1: {10, 11}, 2: {11, 12}, 3: {11}}) == {1, 2}


def test_manual_audit_can_demote_but_cannot_invent_number():
    rows = [{
        "global_id": 7, "status": "confirmed", "predicted_number": 10,
        "decision_rule": "automatic",
    }]
    apply_manual_audit(rows, {"decisions": [{
        "global_id": 7, "decision": "unreadable", "reason": "number not visible",
    }]})
    assert rows[0]["status"] == "unreadable"
    assert rows[0]["predicted_number"] is None

    rows[0].update(status="tentative", predicted_number=10)
    with pytest.raises(ValueError, match="cannot invent"):
        apply_manual_audit(rows, {"decisions": [{
            "global_id": 7, "decision": "confirmed", "number": 11,
        }]})
