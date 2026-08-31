from __future__ import annotations

import json

from analysis_lib.player_card import (
    calculate_player_running, load_confirmed_players, resolve_player_metric_ids,
)


def test_identity_aliases_group_and_nonconfirmed_buckets_are_excluded(tmp_path):
    source = tmp_path / "eligibility.json"
    source.write_text(json.dumps({
        "eligible_confirmed": [
            {"global_id": 3, "team": "yellow", "final_number": 15, "confidence": .9},
            {"global_id": 20, "team": "yellow", "final_number": 15, "confidence": .8},
        ],
        "excluded_conflict": [{"global_id": 7}],
        "excluded_mismatch": [{"global_id": 8}],
        "excluded_unreadable": [{"global_id": 9}],
    }), encoding="utf-8")
    players, audit = load_confirmed_players(source)
    assert players["yellow_15"]["global_ids"] == [3, 20]
    assert audit["confirmed_player_identities"] == 1
    assert audit["excluded_conflict"] == 1


def test_running_rejects_short_speed_spikes_and_excludes_abnormal_speed():
    # Two-frame spikes are typical low-camera foot-point jitter, not a sprint.
    speeds = [1.0] * 10 + [8.0, 8.0] + [1.0] * 10 + [16.0]
    rows = [
        {"global_id": "1", "proc_idx": str(frame), "segment_id": "0",
         "speed_mps": str(speed), "step_distance_m": ".1"}
        for frame, speed in enumerate(speeds)
    ]
    summary, heatmap_rows = calculate_player_running(rows, fps=10)
    assert summary["sprint_count"] == 0
    assert summary["abnormal_speed_rows_excluded"] == 1
    assert summary["max_speed_mps"] < 4.5
    assert summary["tracked_visible_time_sec"] == 2.2
    assert summary["playing_time_sec"] is None
    assert len(heatmap_rows) == 22


def test_running_counts_only_sustained_sprint_bouts():
    # At 10 fps the project rule requires at least 0.5 s = 5 frames.
    speeds = [1.0] * 5 + [5.2] * 6 + [3.0] * 5
    rows = [
        {"global_id": "1", "proc_idx": str(frame), "segment_id": "0",
         "speed_mps": str(speed), "step_distance_m": ".1"}
        for frame, speed in enumerate(speeds)
    ]
    summary, _ = calculate_player_running(rows, fps=10)
    assert summary["sprint_count"] == 1
    assert summary["sprint_min_duration_sec"] == 0.5
    assert summary["max_speed_mps"] == 5.2


def test_spatially_conflicting_same_number_ids_are_not_aggregated():
    player = {
        "global_ids": [5, 25],
        "global_id_confidence": {"5": 0.97, "25": 0.95},
    }
    timeseries = {
        5: [{"proc_idx": str(frame), "x_m_smooth": "1", "y_m_smooth": "1"}
            for frame in range(10)],
        25: [{"proc_idx": str(frame), "x_m_smooth": "10", "y_m_smooth": "1"}
             for frame in range(10)],
    }
    resolution = resolve_player_metric_ids(player, timeseries, fps=10)
    assert resolution["metric_global_ids"] == [5]
    assert resolution["excluded_metric_global_ids"] == [25]
    assert resolution["status"] == "identity_overlap_conflict_canonical_only"
