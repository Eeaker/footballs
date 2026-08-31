from pathlib import Path

import json
import cv2
import numpy as np

from analysis_lib.jersey_numbers import (
    adapt_number_results_csv, conservative_status, load_team_hints, read_mot_deduplicated,
    load_existing_tracklet_crops,
)


def test_mot_duplicate_keeps_highest_confidence(tmp_path: Path):
    mot = tmp_path / "tracking_mot.txt"
    mot.write_text(
        "1,7,10,20,30,60,0.2,-1,-1,-1\n"
        "1,7,11,21,30,60,0.9,-1,-1,-1\n"
        "2,7,12,22,30,60,0.8,-1,-1,-1\n",
        encoding="utf-8",
    )
    rows, duplicates = read_mot_deduplicated(mot)
    assert duplicates == 1
    assert len(rows) == 2
    assert rows[0]["x"] == 11


def test_conservative_gate_requires_independent_full_body_evidence():
    voted = {
        "jersey_number": 24, "votes": 5, "confidence": 0.7,
        "head_confidence": 0.8, "winner_margin": 0.4,
        "full_body_sufficient": True,
        "candidates": [{"jersey_number": 24, "votes": 5}],
    }
    assert conservative_status(voted, {})[0] == "confirmed"
    assert conservative_status({**voted, "votes": 4}, {})[0] == "unreadable"
    assert conservative_status({**voted, "full_body_sufficient": False}, {})[0] == "unreadable"


def test_competing_numbers_become_conflict():
    voted = {
        "jersey_number": 24, "votes": 5, "confidence": 0.52,
        "head_confidence": 0.52, "winner_margin": 0.04,
        "full_body_sufficient": True,
        "candidates": [
            {"jersey_number": 24, "votes": 5},
            {"jersey_number": 21, "votes": 3},
        ],
    }
    status, reason = conservative_status(voted, {})
    assert status == "conflict"
    assert reason == "multiple_numbers_with_independent_support"


def test_team_hints_accept_player_card_verifier_json(tmp_path: Path):
    path = tmp_path / "clip_eligibility.json"
    path.write_text(
        '{"eligible_confirmed":[{"global_id":1,"team":"yellow"}],'
        '"excluded_unreadable":[{"global_id":2,"team":"blue"}]}',
        encoding="utf-8",
    )
    assert load_team_hints(path) == {1: "yellow", 2: "blue"}


def test_v10_csv_adapter_preserves_confirmed_and_unreadable(tmp_path: Path):
    numbers = tmp_path / "jersey_number_results.csv"
    numbers.write_text(
        "global_id,team,predicted_number,confidence,status\n"
        "1,yellow,24,0.9,confirmed\n"
        "2,blue,,0,unreadable\n",
        encoding="utf-8",
    )
    mot = tmp_path / "mot.txt"
    mot.write_text(
        "1,1,0,0,10,20,0.9,-1,-1,-1\n2,2,20,0,10,20,0.9,-1,-1,-1\n",
        encoding="utf-8",
    )
    output = tmp_path / "clip_eligibility.json"
    adapt_number_results_csv(numbers=numbers, mot=mot, output=output)
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["eligible_confirmed"][0]["final_number"] == 24
    assert data["excluded_unreadable"][0]["global_id"] == 2


def test_existing_candidates_can_resume_without_video_decode(tmp_path: Path):
    folder = tmp_path / "candidates" / "gid_007"
    folder.mkdir(parents=True)
    cv2.imwrite(str(folder / "frame_000123.jpg"), np.zeros((40, 20, 3), np.uint8))
    rows, audit = load_existing_tracklet_crops(tmp_path / "candidates")
    assert rows[0]["display_track_id"] == 7
    assert rows[0]["frame"] == 123
    assert audit["written_crops"] == 1
