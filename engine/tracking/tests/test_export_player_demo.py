from export_player_demo import (
    alias_overlap_report, audit_identity_map, clip_bounds, load_identity_map, parse_player,
    select_player_events,
)


def test_load_identity_map_merges_same_team_number(tmp_path):
    path = tmp_path / "map.csv"
    path.write_text(
        "global_id,队伍,号码,归并说明,置信度,备注\n"
        "0,白,20,与ID07为同一人,确定,\n"
        "7,白,20,与ID00为同一人,确定,\n"
        "29,排除,,,排除,场边人员\n",
        encoding="utf-8",
    )
    mapping = load_identity_map(path)
    assert mapping[0]["canonical_key"] == mapping[7]["canonical_key"] == "白_20"
    assert mapping[29]["excluded"] is True
    assert mapping[29]["canonical_key"] is None


def test_select_player_events_prefers_auto_and_multiple_types():
    rows = [
        {"event_id": 1, "primary_global_id": 0, "actor_attribution_status": "auto",
         "base_event_type": "射门_大力踢球", "score": 9.0, "event_time_seconds": 20.0},
        {"event_id": 2, "primary_global_id": 7, "actor_attribution_status": "auto",
         "base_event_type": "关键动作", "score": 7.0, "event_time_seconds": 40.0},
        {"event_id": 3, "primary_global_id": 0, "actor_attribution_status": "review",
         "base_event_type": "传球_解围_方向突变", "score": 99.0, "event_time_seconds": 60.0},
    ]
    selected = select_player_events(rows, {0, 7}, count=2)
    assert {row["event_id"] for row in selected} == {1, 2}


def test_clip_bounds_and_player_parser():
    assert clip_bounds(3.0, 100.0, 15.0, 15.0) == (0.0, 18.0)
    assert clip_bounds(98.0, 100.0, 15.0, 15.0) == (83.0, 100.0)
    assert parse_player("白：20") == ("白", "20")


def test_alias_overlap_report_flags_impossible_same_person_overlap():
    mot = {
        10: [(0, 0, 0, 10, 10), (7, 20, 0, 10, 10)],
        11: [(0, 0, 0, 10, 10)],
    }
    report = alias_overlap_report(mot, {0, 7}, fps=25.0)
    assert report["conflict"] is True
    assert report["overlap_frames"] == 1
    assert report["render_policy"] == "highlight_event_source_global_id_only"


def test_audit_identity_map_counts_conflicting_groups():
    identity = {
        0: {"canonical_key": "白_20"}, 7: {"canonical_key": "白_20"},
        1: {"canonical_key": "黄_2"},
    }
    mot = {10: [(0, 0, 0, 1, 1), (7, 2, 0, 1, 1)]}
    report = audit_identity_map(identity, mot, 25.0)
    assert report["multi_candidate_player_groups"] == 1
    assert report["conflicting_groups"] == 1
