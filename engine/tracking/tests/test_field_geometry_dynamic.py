import numpy as np

from tracking_lib.field_geometry import FieldGeometryProvider


def _provider(keyframes):
    return FieldGeometryProvider({
        "enabled": True,
        "mode": "polygon_keyframes",
        "margin": 0.0,
        "keyframes": keyframes,
    })


def test_same_vertex_count_is_interpolated_between_keyframes():
    provider = _provider([
        {"frame_index": 0, "points": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        {"frame_index": 10, "points": [[10, 0], [20, 0], [20, 10], [10, 10]]},
    ])
    assert np.allclose(provider.polygon_at(5), [[5, 0], [15, 0], [15, 10], [5, 10]])


def test_changed_vertex_count_uses_nearest_complete_keyframe_polygon():
    four = [[0, 0], [10, 0], [10, 10], [0, 10]]
    five = [[10, 5], [15, 0], [20, 0], [20, 10], [10, 10]]
    provider = _provider([
        {"frame_index": 0, "points": four},
        {"frame_index": 10, "points": five},
    ])
    assert np.array_equal(provider.polygon_at(4), np.asarray(four, np.float32))
    assert np.array_equal(provider.polygon_at(6), np.asarray(five, np.float32))
