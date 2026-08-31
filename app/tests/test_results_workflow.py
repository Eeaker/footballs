from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from app.services import results
from app.services import reviews


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["global_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "tracking": root / "tracking",
        "analysis": root / "match_analysis" / "analysis",
        "running": root / "match_analysis" / "metric_running",
        "ocr": root / "number_ocr",
        "cards": root / "player_cards",
        "formal_cards": root / "player_cards_formal",
        "highlights": root / "highlights",
        "events": root / "events_for_annotation.json",
        "report_html": root / "match_report.html",
        "report_pdf": root / "match_report.pdf",
        "replay_video": root / "metric_pitch_replay.mp4",
        "artifact_manifest": root / "artifact_manifest.json",
        "identity_audit": root / "identity_audit",
    }


def test_pitch_keeps_all_technical_ids(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    rows = [
        {"global_id": gid, "proc_idx": 0, "x_m_smooth": gid % 45, "y_m_smooth": gid % 25}
        for gid in range(40)
    ]
    _write_csv(paths["running"] / "player_running_timeseries.csv", rows)
    monkeypatch.setattr(results, "output_paths", lambda project: paths)
    payload = results.pitch_data({"settings": {}})
    assert len(payload["trails"]) == 40


def test_replay_exposes_image_coordinates_ball_and_owner(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_csv(paths["running"] / "player_running_timeseries.csv", [{
        "global_id": 7, "proc_idx": 10, "x_m_smooth": 4, "y_m_smooth": 5,
        "foot_x_px": 640, "foot_y_px": 720, "speed_mps": 2.5,
    }, {
        "global_id": 7, "proc_idx": 11, "x_m_smooth": 4.1, "y_m_smooth": 5.1,
        "foot_x_px": 642, "foot_y_px": 721, "speed_mps": 2.6,
    }])
    _write_csv(paths["analysis"] / "player_team_map.csv", [{"global_id": 7, "team_id": "team_0"}])
    _write_csv(paths["analysis"] / "possession_frame_evidence.csv", [{
        "frame_proc": 10, "global_id": 7, "ball_x_m": 4.1, "ball_y_m": 5.1,
        "player_x_m": 4, "player_y_m": 5,
    }])
    _write_csv(paths["tracking"] / "tracking" / "ball_positions_observed.csv", [{
        "frame_proc": 10, "ball_x_px": 650, "ball_y_px": 710, "observed": 1,
    }])
    mot = paths["tracking"] / "tracking" / "tracking_mot.txt"
    mot.parent.mkdir(parents=True, exist_ok=True)
    mot.write_text("11,7,600,500,80,220,0.9,-1,-1,-1\n12,7,602,501,80,220,0.9,-1,-1,-1\n", encoding="utf-8")
    monkeypatch.setattr(results, "output_paths", lambda project: paths)
    payload = results.replay_data({"video": {"fps": 30, "duration_seconds": 1}, "settings": {}}, 120)
    frame = payload["frames"][0]
    assert frame["players"][0]["image"] == [640.0, 720.0]
    assert frame["players"][0]["bbox"] == [600.0, 500.0, 80.0, 220.0, 0.9]
    assert frame["ball_image"] == [650.0, 710.0]
    assert frame["possession_id"] == 7
    assert payload["frames"][1]["ball_image"] is None
    window = results.replay_data(
        {"video": {"fps": 30, "duration_seconds": 1, "frame_count": 30}, "settings": {}},
        start_frame=10, frame_count=2,
    )
    assert window["sampling_mode"] == "source_frame"
    assert window["frame_step"] == 1
    assert [row["frame"] for row in window["frames"][:2]] == [10, 11]


def test_same_person_key_merges_ids_and_metrics(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_csv(paths["running"] / "player_running_summary.csv", [
        {"global_id": 1, "total_distance_m": 100, "valid_duration_sec": 10, "peak_speed_mps_p95": 4},
        {"global_id": 2, "total_distance_m": 200, "valid_duration_sec": 20, "peak_speed_mps_p95": 6},
    ])
    monkeypatch.setattr(results, "output_paths", lambda project: paths)
    monkeypatch.setattr(
        "app.services.reviews.identity_mapping_dict",
        lambda project: {
            "1": {"name": "张三", "person_key": "roster:3", "jersey_number": "9"},
            "2": {"name": "张三", "person_key": "roster:3", "jersey_number": "9"},
        },
    )
    low = {key: 40 for key in ("speed", "endurance", "running", "passing", "control", "shooting", "defense", "physical")}
    high = {key: 70 for key in low}
    monkeypatch.setattr(
        "app.services.reviews.player_assessment_dict",
        lambda project: {
            "1": {"scores": low, "status": "confirmed", "note": "片段一"},
            "2": {"scores": high, "status": "confirmed", "note": "片段二"},
        },
    )
    rows = results.players({"kind": "analysis"})
    assert len(rows) == 1
    assert rows[0]["global_ids"] == [1, 2]
    assert rows[0]["total_distance_m"] == 300
    assert rows[0]["visible_time_sec"] == 30
    assert rows[0]["max_speed_mps"] == 6
    assert rows[0]["assessment"]["scores"]["speed"] == 60
    assert rows[0]["assessment"]["source"] == "human_time_weighted"


def test_heatmap_image_fallback_merges_linked_ids_by_visible_time(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    for gid, color, seconds in ((1, (255, 0, 0), 10), (2, (0, 0, 255), 30)):
        folder = paths["formal_cards"] / f"unknown_{gid}"
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), color).save(folder / "heatmap.png")
        (folder / "running.json").write_text(
            json.dumps({"summary": {"playing_time_sec": seconds}}), encoding="utf-8"
        )
    monkeypatch.setattr(results, "output_paths", lambda project: paths)
    merged = results.heatmap_image_path({"id": "p"}, [1, 2])
    assert merged is not None and merged.is_file()
    pixel = Image.open(merged).convert("RGB").getpixel((0, 0))
    assert pixel[0] in range(62, 66)
    assert pixel[2] in range(190, 194)


def test_identity_review_links_multiple_ids_to_one_person(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(reviews, "project_dir", lambda project_id: tmp_path)
    monkeypatch.setattr(reviews, "_candidate_global_ids", lambda project: [1, 2, 3])
    state = reviews.save_identity_mapping(
        {"id": "p", "kind": "analysis"}, 1, name="张三", jersey_number="9",
        roster_index=4, linked_global_ids=[1, 2],
    )
    assert state["mappings"]["1"]["person_key"] == "roster:4"
    assert state["mappings"]["2"]["person_key"] == "roster:4"
    assert "3" not in state["mappings"]


def test_identity_merge_rejects_simultaneous_or_other_team_ids(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(reviews, "project_dir", lambda project_id: tmp_path)
    monkeypatch.setattr(reviews, "_candidate_global_ids", lambda project: [1, 2, 3])
    monkeypatch.setattr(
        reviews, "_identity_merge_evidence",
        lambda project: ({1: "team_0", 2: "team_0", 3: "team_1"}, {1: {10, 11}, 2: {11, 12}, 3: {20}}),
    )
    candidates = reviews.identity_merge_candidates({"id": "p", "kind": "analysis"}, 1)["candidates"]
    by_id = {row["global_id"]: row for row in candidates}
    assert not by_id[2]["compatible"]
    assert "同一时间同时出现" in by_id[2]["reasons"][0]
    assert not by_id[3]["compatible"]
    assert "不同球队" in by_id[3]["reasons"][0]
    try:
        reviews.save_identity_mapping(
            {"id": "p", "kind": "analysis"}, 1, name="张三", linked_global_ids=[1, 2]
        )
    except ValueError as exc:
        assert "不能合并" in str(exc)
    else:
        raise AssertionError("simultaneous IDs must not merge")


def test_player_report_annotation_round_trip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(reviews, "project_dir", lambda project_id: tmp_path)
    monkeypatch.setattr(reviews, "_candidate_global_ids", lambda project: [7])
    saved = reviews.save_player_report_annotation(
        {"id": "p", "kind": "analysis"}, 7,
        {"position": "中前卫", "strengths_summary": "接球前观察充分"},
    )
    assert saved["fields"]["position"] == "中前卫"
    assert saved["fields"]["strengths_summary"] == "接球前观察充分"
