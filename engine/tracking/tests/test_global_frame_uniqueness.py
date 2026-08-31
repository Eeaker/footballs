import pytest

from run_tracking import assert_unique_global_frames


def test_global_frame_uniqueness_accepts_disjoint_tracklets():
    assert_unique_global_frames(
        {10: 0, 11: 0},
        {10: {"frames": {1, 2}}, 11: {"frames": {3, 4}}},
    )


def test_global_frame_uniqueness_rejects_one_frame_overlap():
    with pytest.raises(RuntimeError, match="同帧身份唯一性"):
        assert_unique_global_frames(
            {10: 0, 11: 0},
            {10: {"frames": {1, 2}}, 11: {"frames": {2, 3}}},
        )
