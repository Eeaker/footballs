import json

import pytest

from apply_identity_map import load_assignments, remap_events, remap_mot


def test_identity_mapping_merges_candidate_ids(tmp_path):
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps({"assignments": {
        "3": {"player_id": 2, "team": "yellow", "reviewed": True},
        "9": {"player_id": 2, "team": "yellow", "reviewed": True},
    }}), encoding="utf-8")
    assignments = load_assignments(mapping, 16)
    mot = tmp_path / "in.txt"
    mot.write_text("1,3,1,2,3,4,0.9,-1,-1,-1\n2,9,1,2,3,4,0.8,-1,-1,-1\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    report = remap_mot(mot, out, assignments)
    assert report["written_rows"] == 2
    assert [line.split(",")[1] for line in out.read_text().splitlines()] == ["2", "2"]


def test_identity_mapping_rejects_player_outside_roster(tmp_path):
    mapping = tmp_path / "map.json"
    mapping.write_text('{"0": 16}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_assignments(mapping, 16)


def test_unmapped_event_requires_review(tmp_path):
    source = tmp_path / "events.json"
    target = tmp_path / "out.json"
    source.write_text('[{"primary_global_id": 7}]', encoding="utf-8")
    remap_events(source, target, {})
    event = json.loads(target.read_text(encoding="utf-8"))[0]
    assert event["player_id"] is None
    assert event["identity_review_required"] is True
