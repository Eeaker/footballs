from __future__ import annotations

import csv

from analysis_lib.acceptance import evaluate_annotations, make_sample_rows, systematic_sample
from analysis_lib.passes import PassEvent


def event(index: int) -> PassEvent:
    return PassEvent(
        pass_id=index, transition_id=index, from_global_id=1, to_global_id=2,
        team_id="team_0", release_frame_proc=index, receive_frame_proc=index + 1,
        receive_confirmed_frame_proc=index + 3, transfer_gap_frames=0,
        start_x_m=0, start_y_m=0, end_x_m=1, end_y_m=0, dx_m=1, dy_m=0,
        distance_m=1, direction_angle_deg=0, transfer_speed_mps=10,
        classification="active_directed_pass_candidate",
        intent_proxy="stable_A_and_B+same_team+metric_displacement;human_review_required",
    )


def test_systematic_sample_is_deterministic_and_spans_timeline():
    events = [event(index) for index in range(100)]
    first = systematic_sample(events, 20)
    second = systematic_sample(events, 20)
    assert [row.pass_id for row in first] == [row.pass_id for row in second]
    assert len(first) == 20
    assert first[0].pass_id < 5
    assert first[-1].pass_id > 94
    assert len({row.pass_id for row in first}) == 20


def test_twenty_labels_and_eighty_percent_are_both_required(tmp_path):
    sample = make_sample_rows([event(index) for index in range(20)], 20, 10.0)
    annotation = tmp_path / "annotations.csv"
    with annotation.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pass_id", "human_is_pass"])
        writer.writeheader()
        for index in range(20):
            writer.writerow({"pass_id": index, "human_is_pass": "yes" if index < 16 else "no"})
    result = evaluate_annotations(sample, annotation, .8, 20)
    assert result["passed"] is True
    assert result["agreement_rate"] == .8


def test_fewer_than_twenty_events_can_never_pass(tmp_path):
    sample = make_sample_rows([event(index) for index in range(10)], 20, 10.0)
    annotation = tmp_path / "annotations.csv"
    with annotation.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pass_id", "human_is_pass"])
        writer.writeheader()
        for index in range(10):
            writer.writerow({"pass_id": index, "human_is_pass": "yes"})
    result = evaluate_annotations(sample, annotation, .8, 20)
    assert result["passed"] is False
    assert result["available_sample_events"] == 10
