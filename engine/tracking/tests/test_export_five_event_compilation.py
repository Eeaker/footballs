from export_five_event_compilation import frame_bounds, select_balanced_candidates


def test_frame_bounds_are_fifteen_seconds_each_side():
    start, end = frame_bounds(2400, 10000, 24.0, 15.0, 15.0)
    assert start == 2040
    assert end == 2760
    assert end - start == 720


def test_balanced_selection_uses_each_type_before_second_round():
    rows = [
        {"event_id": 1, "event_frame_proc": 10, "base_event_type": "A", "score": 9},
        {"event_id": 2, "event_frame_proc": 20, "base_event_type": "A", "score": 8},
        {"event_id": 3, "event_frame_proc": 30, "base_event_type": "B", "score": 7},
        {"event_id": 4, "event_frame_proc": 40, "base_event_type": "C", "score": 6},
    ]
    selected = select_balanced_candidates(rows, 3)
    assert {row["base_event_type"] for row in selected} == {"A", "B", "C"}
