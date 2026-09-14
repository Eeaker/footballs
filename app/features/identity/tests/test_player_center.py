import csv
import json
from pathlib import Path

from app.services import results


def _write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_player_center_shows_only_merged_identity_with_ocr_label(monkeypatch, tmp_path: Path):
    paths = {
        "root": tmp_path,
        "cards": tmp_path / "cards",
        "running": tmp_path / "running",
        "analysis": tmp_path / "analysis",
        "ocr": tmp_path / "ocr",
    }
    catalog = tmp_path / "identity_resolution" / "filtered"
    catalog.mkdir(parents=True)
    (catalog / "identity_catalog.json").write_text(json.dumps({
        "published_players": [{"canonical_id": 1, "source_ids": [1, 2], "confirmed": True}],
        "isolated_ids": [3], "quarantine_ids": [4],
    }), encoding="utf-8")
    _write(paths["running"] / "player_running_summary.csv", [
        {"global_id": 1, "total_distance_m": 1000, "valid_duration_sec": 20, "peak_speed_mps_p95": 6},
        {"global_id": 3, "total_distance_m": 800, "valid_duration_sec": 18, "peak_speed_mps_p95": 5},
    ])
    _write(paths["analysis"] / "player_team_map.csv", [{"global_id": 1, "team_id": "team_0"}])
    _write(paths["ocr"] / "jersey_number_results.csv", [
        {"global_id": 1, "predicted_number": "25", "status": "confirmed", "team": "蓝队"},
        {"global_id": 3, "predicted_number": "7", "status": "confirmed", "team": "黄队"},
    ])
    monkeypatch.setattr(results, "output_paths", lambda project: paths)
    monkeypatch.setattr("app.services.reviews.identity_mapping_dict", lambda project: {})
    monkeypatch.setattr("app.services.reviews.player_assessment_dict", lambda project: {})
    rows = results.players({"id": "match-1", "kind": "analysis"})
    assert len(rows) == 1
    assert rows[0]["global_ids"] == [1, 2]
    assert rows[0]["player_id"] == "蓝队25号"
    assert rows[0]["avatar_url"].endswith("/players/1/avatar.jpg")

