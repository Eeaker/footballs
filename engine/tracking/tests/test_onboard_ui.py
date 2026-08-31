from onboard.ui import (
    calibration_decrease_step,
    calibration_increase_step,
    calibration_move,
    navigation_direction,
    resizable_window_flags,
    update_keyframe_selection,
    update_point_selection,
    initial_keyframe_state,
    is_enter_key,
    next_unannotated_suggested_frame,
    partition_point_annotations,
    polygon_signed_area,
    validate_closed_annotations,
)


def test_enter_key_is_recognised_across_opencv_backends():
    assert is_enter_key(10)
    assert is_enter_key(13)
    assert is_enter_key(0x10000 | 13)
    assert not is_enter_key(27)


def test_enter_on_first_dynamic_keyframe_advances_to_next_suggestion():
    annotations = {0: [[0, 0], [10, 0], [10, 10], [0, 10]]}
    assert next_unannotated_suggested_frame([0, 500, 999], annotations, 0, 1000) == 500
    annotations[500] = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert next_unannotated_suggested_frame([0, 500, 999], annotations, 500, 1000) == 999


def test_next_suggested_frame_wraps_and_returns_none_when_all_are_annotated():
    assert next_unannotated_suggested_frame([0, 500, 999], {500: [[1, 1]]}, 700, 1000) == 999
    assert next_unannotated_suggested_frame([0, 500, 999], {999: [[1, 1]]}, 999, 1000) == 0
    assert next_unannotated_suggested_frame(
        [0, 500], {0: [[1, 1]], 500: [[2, 2]]}, 0, 1000,
    ) is None


def test_incomplete_points_on_another_frame_are_treated_as_draft_not_failure():
    annotations = {
        0: [[0, 0], [10, 0], [10, 10], [0, 10]],
        24880: [[20, 20], [30, 20], [30, 30], [20, 30]],
        31102: [[99, 99]],
    }
    complete, drafts = partition_point_annotations(annotations, 4, 8)
    assert sorted(complete) == [0, 24880]
    assert drafts == {31102: [[99, 99]]}


def test_calibration_step_sequence_is_exactly_one_then_tens_to_sixty():
    step = 1
    values = []
    for _ in range(7):
        step = calibration_increase_step(step)
        values.append(step)
    assert values == [10, 20, 30, 40, 50, 60, 60]


def test_calibration_step_decreases_to_one_without_zero_or_negative_values():
    assert calibration_decrease_step(60) == 50
    assert calibration_decrease_step(20) == 10
    assert calibration_decrease_step(10) == 1
    assert calibration_decrease_step(1) == 1


def test_navigation_is_frame_exact_and_clamped_at_video_boundaries():
    assert calibration_move(100, 10, +1, 1000) == 110
    assert calibration_move(100, 60, -1, 1000) == 40
    assert calibration_move(3, 60, -1, 1000) == 0
    assert calibration_move(990, 60, +1, 1000) == 999


def test_arrow_key_codes_are_supported_on_windows_linux_and_legacy_backends():
    for key in (2555904, 65363, 0x270000, 83):
        assert navigation_direction(key) == 1
    for key in (2424832, 65361, 0x250000, 81):
        assert navigation_direction(key) == -1
    assert navigation_direction(ord("j")) == 0


def test_calibration_windows_allow_horizontal_and_vertical_resize():
    import cv2

    assert resizable_window_flags() & cv2.WINDOW_FREERATIO


def test_mouse_click_adds_and_right_click_removes_current_keyframe_idempotently():
    selected = [0, 100]
    assert update_keyframe_selection(selected, 50, add=True) == [0, 50, 100]
    assert update_keyframe_selection(selected, 50, add=True) == [0, 50, 100]
    assert update_keyframe_selection(selected, 100, add=False) == [0, 50]
    assert update_keyframe_selection(selected, 999, add=False) == [0, 50]


def test_reference_point_mouse_callback_adds_undoes_and_honours_limit():
    import cv2

    points = []
    update_point_selection(points, cv2.EVENT_LBUTTONDOWN, 120, 240, maximum=2)
    update_point_selection(points, cv2.EVENT_LBUTTONDOWN, 300, 400, maximum=2)
    update_point_selection(points, cv2.EVENT_LBUTTONDOWN, 500, 600, maximum=2)
    assert points == [[120.0, 240.0], [300.0, 400.0]]
    update_point_selection(points, cv2.EVENT_RBUTTONDOWN, 0, 0, maximum=2)
    assert points == [[120.0, 240.0]]


def test_suggested_frames_set_initial_position_but_are_not_silently_preselected():
    position, selected = initial_keyframe_state([100, 500, 900], total=1000)
    assert position == 100
    assert selected == []


def test_suggested_frames_are_clamped_and_empty_input_starts_at_zero():
    assert initial_keyframe_state([-20, 1200], total=1000) == (0, [])
    assert initial_keyframe_state([], total=1000) == (0, [])


def test_four_ordered_field_corners_form_a_valid_closed_quadrilateral():
    annotations = {100: [[10, 10], [210, 20], [190, 140], [20, 150]]}
    valid, _ = validate_closed_annotations(annotations, expected_points=4, min_area_px=500)
    assert valid
    assert abs(polygon_signed_area(annotations[100])) > 500


def test_visible_field_polygon_accepts_five_to_eight_vertices():
    annotations = {
        100: [[10, 70], [70, 20], [210, 20], [210, 150], [10, 150]],
        200: [[15, 75], [75, 25], [215, 25], [215, 155], [15, 155]],
    }
    valid, reason = validate_closed_annotations(
        annotations, minimum_points=4, maximum_points=8, min_area_px=500,
    )
    assert valid, reason


def test_dynamic_visible_field_allows_vertex_count_to_change_between_keyframes():
    annotations = {
        100: [[10, 10], [210, 10], [210, 150], [10, 150]],
        200: [[15, 70], [70, 20], [215, 20], [215, 155], [15, 155]],
    }
    valid, reason = validate_closed_annotations(
        annotations, minimum_points=4, maximum_points=8, min_area_px=500,
    )
    assert valid, reason


def test_visible_field_polygon_rejects_more_than_eight_vertices():
    points = [[10, 10], [100, 10], [200, 10], [220, 60], [220, 140],
              [120, 160], [20, 160], [10, 100], [5, 50]]
    valid, reason = validate_closed_annotations(
        {100: points}, minimum_points=4, maximum_points=8, min_area_px=500,
    )
    assert not valid
    assert "outside 4-8" in reason


def test_self_crossing_four_points_are_rejected():
    annotations = {100: [[10, 10], [200, 150], [200, 10], [10, 150]]}
    valid, reason = validate_closed_annotations(annotations, expected_points=4, min_area_px=500)
    assert not valid
    assert "self-intersects" in reason


def test_incomplete_or_opposite_direction_dynamic_polygons_are_rejected():
    incomplete = {100: [[10, 10], [200, 10], [200, 150]]}
    assert not validate_closed_annotations(incomplete, 4, 500)[0]
    annotations = {
        100: [[10, 10], [200, 10], [200, 150], [10, 150]],
        200: [[20, 160], [210, 160], [210, 20], [20, 20]],
    }
    valid, reason = validate_closed_annotations(annotations, 4, 500)
    assert not valid
    assert "direction" in reason
