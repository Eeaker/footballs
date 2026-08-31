import json
from types import SimpleNamespace

import numpy as np

import pipeline


def _args(tmp_path, *, vid_stride=2):
    return SimpleNamespace(
        vid_stride=vid_stride,
        ball_max_gap=30,
        edge_margin=0.0,
        event_min_gap=2.0,
        event_percentile=92.0,
        pre_sec=3.0,
        post_sec=2.0,
        n_clips=0,
        outdir=str(tmp_path),
    )


def test_stage4_uses_stage1_processed_frame_count_without_second_stride(tmp_path):
    """A late processed-frame detection must fit in Stage 4 at vid_stride > 1."""
    args = _args(tmp_path)
    detections = [
        (8, 7, 0.0, 0.0, 10.0, 10.0, 0.9),
        (9, 7, 1.0, 0.0, 10.0, 10.0, 0.9),
    ]
    ball_pos = {i: (float(i), 0.0, 0.9) for i in range(10)}

    pipeline.stage4_events(
        args,
        detections,
        ball_pos,
        {7: 0},
        total_frames=10,
        fps=30.0,
        out_fps=15.0,
    )

    rows = json.loads((tmp_path / "event_index.json").read_text(encoding="utf-8"))
    assert all(0 <= row["event_frame_proc"] < 10 for row in rows)


def test_raw_frame_sampling_matches_ultralytics_video_loader():
    """Ultralytics grabs V frames before retrieving raw frame V-1."""
    assert [i for i in range(8) if pipeline.is_sampled_raw_frame(i, 2)] == [1, 3, 5, 7]
    assert pipeline.raw_frame_index_for_proc(0, 2) == 1
    assert pipeline.raw_frame_index_for_proc(3, 2) == 7


def test_processed_fps_is_raw_fps_divided_once():
    assert np.isclose(pipeline.processed_fps(30.0, 2), 15.0)


def test_reid_interval_stays_constant_when_video_stride_changes():
    assert pipeline.resolve_reid_stride(30.0, 1, 10, 0.33) == 10
    assert pipeline.resolve_reid_stride(30.0, 2, 10, 0.33) == 5
    assert pipeline.resolve_reid_stride(30.0, 5, 10, 0.33) == 2
    assert pipeline.resolve_reid_stride(30.0, 2, 7, 0) == 7


def _tracklet(frames):
    return {
        "frames": set(frames),
        "emb_sum": np.ones(2048, dtype=np.float32),
        "emb_cnt": 1,
        "col_sum": np.ones(256, dtype=np.float32),
        "col_cnt": 1,
        "first": min(frames),
        "last": max(frames),
    }


def test_presence_ratio_is_hard_filter_by_default():
    tracklets = {
        1: _tracklet(range(50)),
        2: _tracklet(range(5)),
    }
    args = SimpleNamespace(
        max_ids=35,
        min_track_frames=1,
        color_min=0.15,
        wa=0.6,
        wc=0.4,
        merge_floor=0.1,
        min_presence_ratio=0.10,
        allow_presence_backfill=False,
    )
    mapping = pipeline.stage2_global_reassoc(tracklets, total_frames=100, args=args)
    assert 1 in mapping
    assert 2 not in mapping


def test_presence_backfill_requires_explicit_opt_in():
    tracklets = {
        1: _tracklet(range(50)),
        2: _tracklet(range(5)),
    }
    args = SimpleNamespace(
        max_ids=35,
        min_track_frames=1,
        color_min=0.15,
        wa=0.6,
        wc=0.4,
        merge_floor=0.1,
        min_presence_ratio=0.10,
        allow_presence_backfill=True,
    )
    mapping = pipeline.stage2_global_reassoc(tracklets, total_frames=100, args=args)
    assert set(mapping) == {1, 2}


def test_clip_bounds_are_exactly_five_seconds_in_processed_domain():
    start, end, end_exclusive = pipeline.clip_bounds(
        event_frame=150, proc_total=1000, out_fps=15.0,
        pre_sec=3.0, post_sec=2.0,
    )
    assert start == 105
    assert end == 179
    assert end_exclusive == 180
    assert end - start + 1 == 75
