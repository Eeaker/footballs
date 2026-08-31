from onboard.ui import polygon_quality


def test_small_incomplete_pitch_polygon_is_rejected():
    points = [[1782, 638], [417, 677], [389, 920], [850, 801]]
    quality = polygon_quality(points, (1080, 1920, 3))
    assert quality["area_ratio"] < .10
    assert not quality["valid"]


def test_broad_convex_pitch_polygon_is_accepted():
    points = [[100, 250], [1820, 250], [1910, 1030], [10, 1030]]
    quality = polygon_quality(points, (1080, 1920, 3))
    assert quality["valid"]
