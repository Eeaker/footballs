import numpy as np

from onboard.calibration import compute_homography, project_points, validate_segments


def test_homography_and_independent_segment_validation():
    image = [[10, 20], [110, 20], [110, 70], [10, 70]]
    world = [[0, 0], [100, 0], [100, 50], [0, 50]]
    homography, rmse = compute_homography(image, world)
    assert rmse < 1e-4
    projected = project_points(homography, [[35, 45], [85, 45]])
    assert np.allclose(projected, [[25, 25], [75, 25]], atol=1e-3)
    assert validate_segments(homography, [{"p1": [35, 45], "p2": [85, 45], "length_m": 50}]) < 1e-3

