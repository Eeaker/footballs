import numpy as np

from tracking_lib.actor import BoxObservation, attribute_actor, interpolate_ball
from run_tracking import export_ball_positions


def test_interpolation_rejects_long_gap():
    bx, by, reliable = interpolate_ball({0: (0, 0), 2: (2, 4), 8: (8, 8)}, 10, max_gap=3)
    assert reliable[:3].all()
    assert bx[1] == 1 and by[1] == 2
    assert not reliable[3:8].any()


def test_actor_nearest_to_ball_before_event():
    ball_x = np.full(20, 100.0)
    ball_y = np.full(20, 100.0)
    reliable = np.ones(20, dtype=bool)
    boxes = {
        9: [
            BoxObservation(9, 3, 85, 60, 30, 45),
            BoxObservation(9, 8, 260, 60, 30, 45),
        ],
        10: [
            BoxObservation(10, 3, 88, 60, 30, 45),
            BoxObservation(10, 8, 250, 60, 30, 45),
        ],
    }
    result = attribute_actor(10, 10.0, boxes, ball_x, ball_y, reliable)
    assert result["primary_global_id"] == 3
    assert result["actor_candidates"][0]["global_id"] == 3


def test_actor_requires_review_without_ball():
    result = attribute_actor(
        5, 10.0, {}, np.full(10, np.nan), np.full(10, np.nan), np.zeros(10, dtype=bool)
    )
    assert result["primary_global_id"] is None
    assert result["actor_attribution_status"] == "review"


def test_export_ball_positions_accepts_optional_confidence(tmp_path):
    output = tmp_path / "ball.csv"
    export_ball_positions(output, {7: (12.5, 20.25, 0.91)})
    assert output.read_text(encoding="utf-8").splitlines() == [
        "frame_proc,ball_x_px,ball_y_px,observed",
        "7,12.5,20.25,1",
    ]
