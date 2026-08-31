from build_machine_events import _stratified_stage4


def test_stage4_sampling_preserves_original_round_robin_policy():
    rows = [
        {"event_id": 1, "event_type": "b", "score": 1.0},
        {"event_id": 2, "event_type": "a", "score": 2.0},
        {"event_id": 3, "event_type": "a", "score": 3.0},
        {"event_id": 4, "event_type": "b", "score": 4.0},
        {"event_id": 5, "event_type": "c", "score": 5.0},
    ]

    selected = _stratified_stage4(rows, 4)

    # Selection order is a3, b4, c5, a2 and the public result is chronological.
    assert [row["event_id"] for row in selected] == [2, 3, 4, 5]


def test_stage4_sampling_never_duplicates_or_exceeds_available_rows():
    rows = [{"event_id": 1, "event_type": "only", "score": 1.0}]
    assert _stratified_stage4(rows, 20) == rows


def test_stage4_sampling_accepts_tracking_enriched_event_contract():
    rows = [{"event_id": 1, "base_event_type": "关键动作", "score": 1.0}]
    assert _stratified_stage4(rows, 20) == rows
