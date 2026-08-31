from generate_review_manifest import build_rows


def test_review_rows_are_pending_and_unscored():
    rows = build_rows([
        {"event_id": 3, "event_time": "00:42.000", "base_event_type": "关键动作",
         "primary_global_id": 7, "actor_attribution_status": "auto",
         "actor_attribution_reason": "nearest"}
    ], [{"event_id": 3, "clip_file": "event_0003_gid_007.mp4"}])
    assert rows[0]["candidate_global_id"] == 7
    assert rows[0]["review_status"] == "pending"
    assert rows[0]["candidate_labels"] == []
    assert "score" not in rows[0]
