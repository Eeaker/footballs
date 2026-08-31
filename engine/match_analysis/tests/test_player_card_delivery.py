from __future__ import annotations

import json

from export_player_card_delivery_v1 import load_unknown_players


def test_unknown_selection_includes_all_unresolved_players(tmp_path):
    numbers = tmp_path / "numbers.json"
    numbers.write_text(json.dumps({
        "excluded_unreadable": [
            {"global_id": 2, "team": "white", "final_number": "19"},
            {"global_id": 6, "team": "yellow", "final_number": ""},
            {"global_id": 29, "team": "exclude", "final_number": ""},
        ],
        "excluded_conflict": [{"global_id": 7, "team": "white"}],
        "excluded_mismatch": [{"global_id": 12, "team": "yellow"}],
    }), encoding="utf-8")

    assert load_unknown_players(numbers) == [
        {"player_id": "unknown_2", "global_id": 2, "team": "white", "status": "unreadable"},
        {"player_id": "unknown_6", "global_id": 6, "team": "yellow", "status": "unreadable"},
        {"player_id": "unknown_7", "global_id": 7, "team": "white", "status": "conflict"},
        {"player_id": "unknown_12", "global_id": 12, "team": "yellow", "status": "mismatch"},
    ]


def test_unknown_selection_keeps_run_local_three_cluster_team_labels(tmp_path):
    numbers = tmp_path / "numbers.json"
    numbers.write_text(json.dumps({
        "excluded_unreadable": [
            {"global_id": 1, "team": "team_1"},
            {"global_id": 9, "team": "team_0"},
            {"global_id": 29, "team": "exclude"},
        ],
        "excluded_conflict": [{"global_id": 31, "team": "team_2"}],
        "excluded_mismatch": [],
    }), encoding="utf-8")

    assert load_unknown_players(numbers) == [
        {"player_id": "unknown_1", "global_id": 1, "team": "team_1", "status": "unreadable"},
        {"player_id": "unknown_9", "global_id": 9, "team": "team_0", "status": "unreadable"},
        {"player_id": "unknown_31", "global_id": 31, "team": "team_2", "status": "conflict"},
    ]
