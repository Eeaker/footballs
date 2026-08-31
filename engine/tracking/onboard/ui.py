from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def resizable_window_flags() -> int:
    """返回允许横向和纵向独立缩放的 OpenCV 窗口标志。"""
    return int(cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)


def open_resizable_window(title: str, image_shape: tuple[int, ...] | None = None) -> None:
    """创建可双向拉伸的窗口，并按画面尺寸设置不过大的初始窗口。"""
    cv2.namedWindow(title, resizable_window_flags())
    try:
        cv2.setWindowProperty(title, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
    except cv2.error:
        pass
    if image_shape and len(image_shape) >= 2:
        height, width = int(image_shape[0]), int(image_shape[1])
        scale = min(1.0, 1280.0 / max(width, 1), 800.0 / max(height, 1))
        cv2.resizeWindow(title, max(480, int(width * scale)), max(320, int(height * scale)))


def calibration_increase_step(step: int) -> int:
    """按 J 增大精确选帧步长：1→10→20…→60。"""
    return 10 if int(step) < 10 else min(60, int(step) + 10)


def calibration_decrease_step(step: int) -> int:
    """按 K 减小精确选帧步长：60…20→10→1，永不小于1。"""
    return 1 if int(step) <= 10 else max(10, int(step) - 10)


def calibration_move(position: int, step: int, direction: int, total: int) -> int:
    """按当前步长精确前后移动，并把结果限制在有效视频帧范围内。"""
    if total <= 0:
        return 0
    delta = max(1, int(step)) * (1 if direction > 0 else -1)
    return int(np.clip(int(position) + delta, 0, total - 1))


def navigation_direction(key: int) -> int:
    """兼容 Windows、Linux/Qt 和部分旧 OpenCV 后端的左右方向键码。"""
    right_keys = {2555904, 65363, 0x270000, 83}
    left_keys = {2424832, 65361, 0x250000, 81}
    if key in right_keys:
        return 1
    if key in left_keys:
        return -1
    return 0


def is_enter_key(key: int) -> bool:
    """兼容 OpenCV 不同后端返回的 Enter/Return 键码。"""
    return key >= 0 and (key in (10, 13) or (key & 0xFF) in (10, 13))


def update_keyframe_selection(selected: list[int], position: int, add: bool) -> list[int]:
    """幂等地添加或移除关键帧，避免长按/双击产生重复帧。"""
    values = set(int(item) for item in selected)
    if add:
        values.add(int(position))
    else:
        values.discard(int(position))
    selected[:] = sorted(values)
    return selected


def initial_keyframe_state(suggested: list[int], total: int) -> tuple[int, list[int]]:
    """建议帧只决定首次定位，不静默预选，保证鼠标点击产生可见状态变化。"""
    if total <= 0:
        return 0, []
    positions = sorted(set(int(np.clip(frame, 0, total - 1)) for frame in suggested))
    return (positions[0] if positions else 0), []


def next_unannotated_suggested_frame(suggested: list[int], annotations: dict[int, list],
                                     current: int, total: int) -> int | None:
    """返回下一个尚未标注的建议帧；先向后搜索，再从开头循环一次。"""
    if total <= 0:
        return None
    positions = sorted(set(int(np.clip(frame, 0, total - 1)) for frame in suggested))
    remaining = [frame for frame in positions if frame not in annotations]
    if not remaining:
        return None
    return next((frame for frame in remaining if frame > int(current)), remaining[0])


def partition_point_annotations(annotations: dict[int, list[list[float]]],
                                minimum: int, maximum: int
                                ) -> tuple[dict[int, list[list[float]]],
                                           dict[int, list[list[float]]]]:
    """把完整关键帧与1~N个点的未完成草稿分开，防止误点拖垮有效标定。"""
    complete = {index: points for index, points in annotations.items()
                if int(minimum) <= len(points) <= int(maximum)}
    drafts = {index: points for index, points in annotations.items()
              if not int(minimum) <= len(points) <= int(maximum)}
    return complete, drafts


def polygon_signed_area(points: list[list[float]]) -> float:
    """计算按点击顺序构成多边形的有向面积；符号表示顺/逆时针方向。"""
    if len(points) < 3:
        return 0.0
    array = np.asarray(points, np.float64)
    return float(.5 * np.sum(array[:, 0] * np.roll(array[:, 1], -1)
                             - array[:, 1] * np.roll(array[:, 0], -1)))


def _segments_intersect(a, b, c, d) -> bool:
    """判断两条闭线段是否相交，用于拒绝蝴蝶形场地四边形。"""
    def cross(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
                and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    values = (cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b))
    if values[0] * values[1] < 0 and values[2] * values[3] < 0:
        return True
    return ((values[0] == 0 and on_segment(a, c, b))
            or (values[1] == 0 and on_segment(a, d, b))
            or (values[2] == 0 and on_segment(c, a, d))
            or (values[3] == 0 and on_segment(c, b, d)))


def _polygon_self_intersects(points: list[list[float]]) -> bool:
    count = len(points)
    for first in range(count):
        a, b = points[first], points[(first + 1) % count]
        for second in range(first + 1, count):
            if second in {first, (first + 1) % count} or (second + 1) % count == first:
                continue
            if _segments_intersect(a, b, points[second], points[(second + 1) % count]):
                return True
    return False


def polygon_quality(points: list[list[float]], frame_shape: tuple[int, ...],
                    min_area_ratio: float = 0.10) -> dict[str, float | bool | str]:
    """评估可见球场多边形是否覆盖了足够画面，供交互提示和回归检查使用。"""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    frame_area = max(1.0, float(height * width))
    area_ratio = abs(polygon_signed_area(points)) / frame_area if len(points) >= 3 else 0.0
    self_intersects = len(points) >= 4 and _polygon_self_intersects(points)
    valid = (4 <= len(points) <= 8 and not self_intersects
             and area_ratio >= float(min_area_ratio))
    reason = ("valid" if valid else "polygon self-intersects" if self_intersects
              else f"visible-field area ratio {area_ratio:.3f} < {min_area_ratio:.3f}"
              if area_ratio < float(min_area_ratio) else "point count outside 4-8")
    return {"valid": valid, "area_ratio": area_ratio,
            "self_intersects": self_intersects, "reason": reason}


def validate_closed_annotations(annotations: dict[int, list[list[float]]],
                                expected_points: int | None = None,
                                min_area_px: float = 500.0,
                                *, minimum_points: int = 4,
                                maximum_points: int = 8) -> tuple[bool, str]:
    """验证动态可见场地多边形：允许4~8点，并检查面积、自交和点序方向。

    ``expected_points`` 仅为兼容旧调用；提供时仍执行固定点数检查。动态场地
    应使用 ``minimum_points``/``maximum_points``，因为矩形球场投影与画面边界
    相交后可能得到不同顶点数的可见多边形。
    """
    if not annotations:
        return False, "no annotated frame"
    if expected_points is not None:
        minimum_points = maximum_points = int(expected_points)
    minimum_points = max(3, int(minimum_points))
    maximum_points = max(minimum_points, int(maximum_points))
    directions = []
    for frame_index, points in sorted(annotations.items()):
        if not minimum_points <= len(points) <= maximum_points:
            return False, (f"frame {frame_index} has {len(points)} points, "
                           f"outside {minimum_points}-{maximum_points}")
        if _polygon_self_intersects(points):
            return False, f"frame {frame_index} polygon self-intersects"
        area = polygon_signed_area(points)
        if abs(area) < float(min_area_px):
            return False, f"frame {frame_index} polygon area is too small"
        directions.append(1 if area > 0 else -1)
    if len(set(directions)) > 1:
        return False, "polygon point direction differs between keyframes"
    return True, "valid closed visible-field polygons"


def update_point_selection(points: list[list[float]], event: int, x: int, y: int,
                           maximum: int) -> list[list[float]]:
    """处理参照点鼠标事件；左键添加、右键撤销，并限制点数和非法坐标。"""
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < max(1, int(maximum)) and x >= 0 and y >= 0:
        points.append([float(x), float(y)])
    elif event == cv2.EVENT_RBUTTONDOWN and points:
        points.pop()
    return points


def choose_cluster_count(board_path: str | Path, recommended: int) -> int:
    """保持窗口事件循环显示色板；新适配固定为语义三簇。"""
    image = cv2.imread(str(board_path))
    if image is None:
        raise RuntimeError(f"无法读取队色色板: {board_path}")
    footer = np.full((58, image.shape[1], 3), 30, np.uint8)
    cv2.putText(footer, "Two teams + one reject | Enter confirms | Esc cancels",
                (15, 37), cv2.FONT_HERSHEY_SIMPLEX, .68, (240, 240, 240), 2)
    display = np.vstack([image, footer])
    title = "Team color clusters"
    open_resizable_window(title, display.shape)
    while True:
        cv2.imshow(title, display)
        key = cv2.waitKey(30) & 0xFF
        if is_enter_key(key):
            selected = int(recommended); break
        if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow(title)
            raise KeyboardInterrupt("用户取消队色确认")
    cv2.destroyWindow(title)
    return selected


def select_points(frame: np.ndarray, title: str, minimum: int = 4, maximum: int = 8) -> list[list[float]]:
    """鼠标选取点：左键添加，右键/U 撤销，R 清空，Enter 完成，Esc 取消。"""
    points: list[list[float]] = []

    def mouse(event, x, y, _flags, _param):
        update_point_selection(points, event, x, y, maximum)

    open_resizable_window(title, frame.shape)
    cv2.setMouseCallback(title, mouse)
    while True:
        canvas = frame.copy()
        for number, (x, y) in enumerate(points, 1):
            cv2.circle(canvas, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(canvas, str(number), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 255), 2)
        cv2.putText(canvas, f"points {len(points)}/{minimum}-{maximum} | U undo R reset Enter done Esc cancel",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 230, 255), 2)
        cv2.imshow(title, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("u"), 8) and points: points.pop()
        elif key == ord("r"): points.clear()
        elif is_enter_key(key) and minimum <= len(points) <= maximum: break
        elif key == 27:
            points.clear(); break
    cv2.destroyWindow(title)
    return points


def annotate_video_keyframes(video_path: str | Path, suggested: list[int], title: str,
                             minimum: int, maximum: int,
                             close_shape: bool = False,
                             minimum_keyframes: int = 1) -> list[dict]:
    """在精确视频帧点选参照点；场地模式实时闭合并验证4~8点多边形。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法读取视频: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"视频元数据非法，无法标定: {video_path}")
    position, _ = initial_keyframe_state(suggested, total)
    step, annotations = 1, {}
    scrub = {"changed": False, "value": position, "programmatic": False}
    mouse_events: list[tuple[int, int, int]] = []
    status = f"Left click {minimum} point(s) directly on this frame"

    def on_scrub(value: int) -> None:
        if not scrub["programmatic"]:
            scrub["value"] = int(value); scrub["changed"] = True

    def on_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            mouse_events.append((event, int(x), int(y)))

    open_resizable_window(title, (height, width))
    cv2.createTrackbar("frame", title, position, max(total - 1, 1), on_scrub)
    cv2.setMouseCallback(title, on_mouse)
    loaded_position, frame = None, None
    while True:
        if scrub["changed"]:
            position = int(np.clip(scrub["value"], 0, total - 1))
            loaded_position = None; scrub["changed"] = False
            status = f"Exact frame {position}; click points directly on the field"
        while mouse_events:
            event, x, y = mouse_events.pop(0)
            points = annotations.setdefault(position, [])
            before = len(points)
            update_point_selection(points, event, x, y, maximum)
            if not points:
                annotations.pop(position, None)
            if event == cv2.EVENT_LBUTTONDOWN:
                status = (f"Frame {position}: point {len(points)}/{minimum} added"
                          if len(points) > before else f"Frame {position}: maximum {maximum} points reached")
            else:
                status = f"Frame {position}: last point removed"
        if loaded_position != position:
            decoder_position = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            forward_skip = position - decoder_position
            if 0 < forward_skip <= 8:
                for _ in range(forward_skip):
                    if not cap.grab(): break
            elif decoder_position != position:
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = cap.read()
            if not ok: break
            loaded_position = position
        canvas = frame.copy()
        points = annotations.get(position, [])
        current_valid = minimum <= len(points) <= maximum
        validity_reason = ""
        if close_shape and points:
            current_valid, validity_reason = validate_closed_annotations(
                {position: points}, min_area_px=max(500.0, width * height * .10),
                minimum_points=minimum, maximum_points=maximum)
        if len(points) >= 2:
            poly = np.asarray(points, np.int32).reshape((-1, 1, 2))
            line_color = (30, 210, 30) if current_valid else (0, 210, 255)
            cv2.polylines(canvas, [poly], close_shape and len(points) >= minimum, line_color, 3)
            if close_shape and current_valid:
                overlay = canvas.copy(); cv2.fillPoly(overlay, [poly], (30, 190, 30))
                canvas = cv2.addWeighted(overlay, .18, canvas, .82, 0)
        for number, (x, y) in enumerate(points, 1):
            cv2.circle(canvas, (int(x), int(y)), 7, (0, 0, 255), -1)
            cv2.putText(canvas, str(number), (int(x) + 9, int(y) - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
        complete_annotations, draft_annotations = partition_point_annotations(
            annotations, minimum, maximum,
        )
        ready_frames = len(complete_annotations)
        cv2.putText(canvas, f"EXACT FRAME {position}/{total-1}  {position/fps:.3f}s  step={step}",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 230, 255), 2)
        cv2.putText(canvas, "Left click point | Right click undo | R clear frame | Left/Right move | J/K step",
                    (20, 62), cv2.FONT_HERSHEY_SIMPLEX, .52, (20, 230, 255), 2)
        enter_hint = ("Enter: save and next suggested frame"
                      if ready_frames < minimum_keyframes else "Enter: finish calibration")
        cv2.putText(canvas, f"Current points {len(points)}/{minimum}-{maximum} | frames {ready_frames}/{minimum_keyframes}+ | drafts {len(draft_annotations)} | {enter_hint}",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, .52, (20, 230, 255), 2)
        status_color = (30, 30, 235) if status.startswith("Cannot finish") else (60, 220, 60)
        cv2.putText(canvas, status, (20, 118), cv2.FONT_HERSHEY_SIMPLEX, .48, status_color, 2)
        badge = (f"POLYGON READY ({len(points)} POINTS)" if close_shape and current_valid else
                 "KEYFRAME READY - PRESS ENTER" if current_valid else
                 f"NEED {minimum - len(points)} MORE POINT(S)" if len(points) < minimum else "INVALID POINT SET")
        badge_color = (40, 190, 40) if current_valid else (30, 30, 230)
        badge_x = max(10, canvas.shape[1] - 350)
        cv2.rectangle(canvas, (badge_x, 18), (canvas.shape[1] - 12, 55), (245, 245, 245), -1)
        cv2.putText(canvas, badge, (badge_x + 8, 45), cv2.FONT_HERSHEY_SIMPLEX, .56, badge_color, 2)
        if validity_reason and not current_valid and len(points) >= minimum:
            cv2.putText(canvas, validity_reason, (20, 146), cv2.FONT_HERSHEY_SIMPLEX,
                        .48, (30, 30, 230), 2)
        cv2.imshow(title, canvas)
        if cv2.getTrackbarPos("frame", title) != position:
            scrub["programmatic"] = True; cv2.setTrackbarPos("frame", title, position)
            scrub["programmatic"] = False
        key = cv2.waitKeyEx(30); direction = navigation_direction(key)
        if direction:
            position = calibration_move(position, step, direction, total); loaded_position = None
            status = f"Moved to exact frame {position}; existing points are preserved per frame"
        elif key in (ord("j"), ord("J")):
            step = calibration_increase_step(step); status = f"Frame step is now {step}"
        elif key in (ord("k"), ord("K")):
            step = calibration_decrease_step(step); status = f"Frame step is now {step}"
        elif key in (ord("r"), ord("R")):
            annotations.pop(position, None); status = f"Cleared all points on frame {position}"
        elif is_enter_key(key):
            complete_annotations, draft_annotations = partition_point_annotations(
                annotations, minimum, maximum,
            )
            if position in draft_annotations:
                valid = False
                reason = (f"current frame {position} is an incomplete draft "
                          f"({len(draft_annotations[position])}/{minimum} points); add points or press R")
            if close_shape:
                if position not in draft_annotations:
                    valid, reason = validate_closed_annotations(
                        complete_annotations, min_area_px=max(500.0, width * height * .10),
                        minimum_points=minimum, maximum_points=maximum)
            elif position not in draft_annotations:
                valid = bool(complete_annotations)
                reason = "no complete annotated frame"
            required = max(1, int(minimum_keyframes))
            if valid and len(complete_annotations) < required:
                next_frame = next_unannotated_suggested_frame(
                    suggested, complete_annotations, position, total,
                )
                if next_frame is not None:
                    previous = position
                    position = next_frame
                    loaded_position = None
                    status = (f"Saved keyframe {previous}; moved to suggested frame {position}. "
                              f"Add {minimum}-{maximum} points, then press Enter again")
                    continue
                valid = False
                reason = (f"need at least {required} annotated keyframes; "
                          "move to another frame and add points")
            if valid:
                discarded = sorted(draft_annotations)
                for frame_index in discarded:
                    annotations.pop(frame_index, None)
                status = "Calibration points accepted; continue in the terminal"
                break
            status = f"Cannot finish: {reason}"
        elif key == 27:
            annotations.clear(); break
        elif cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            annotations.clear(); break
    cap.release(); cv2.destroyWindow(title)
    return [{"frame_index": int(index), "points": points}
            for index, points in sorted(annotations.items())]


def browse_video_for_keyframes(video_path: str | Path, suggested: list[int],
                               title: str = "Calibration keyframes") -> list[int]:
    """逐帧浏览动态标定位置；方向键移动，J/K调整1~60帧步长。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法读取视频: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        cap.release()
        raise RuntimeError(f"视频没有可浏览帧: {video_path}")
    position, selected = initial_keyframe_state(suggested, total)
    step = 1
    scrub = {"changed": False, "value": position, "programmatic": False}
    status = "No frame selected yet: left click the image to add the current exact frame"

    def on_scrub(value: int) -> None:
        if not scrub["programmatic"]:
            scrub["value"] = int(value)
            scrub["changed"] = True

    mouse_action = {"add": False, "remove": False}

    def on_mouse(event: int, _x: int, _y: int, _flags: int, _param) -> None:
        # 回调只记录动作，统一在主事件循环修改状态，避免不同 HighGUI 后端的线程差异。
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_action["add"] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            mouse_action["remove"] = True

    open_resizable_window(title, (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
    cv2.createTrackbar("frame", title, position, max(total - 1, 1), on_scrub)
    cv2.setMouseCallback(title, on_mouse)
    loaded_position, frame = None, None
    while True:
        if mouse_action["add"]:
            existed = position in selected
            update_keyframe_selection(selected, position, add=True)
            status = (f"Frame {position} was already selected"
                      if existed else f"Added exact frame {position}; Enter continues to point selection")
            mouse_action["add"] = False
        if mouse_action["remove"]:
            existed = position in selected
            update_keyframe_selection(selected, position, add=False)
            status = f"Removed frame {position}" if existed else f"Frame {position} was not selected"
            mouse_action["remove"] = False
        if scrub["changed"]:
            position = int(np.clip(scrub["value"], 0, total - 1))
            loaded_position = None
            scrub["changed"] = False
            status = f"Scrub applied exactly at frame {position}"
        if loaded_position != position:
            decoder_position = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            forward_skip = position - decoder_position
            if 0 < forward_skip <= 8:
                for _ in range(forward_skip):
                    if not cap.grab():
                        break
            elif decoder_position != position:
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = cap.read()
            if not ok:
                break
            loaded_position = position
        canvas = frame.copy()
        is_selected = position in selected
        selected_times = ", ".join(f"{item / fps:.1f}s" for item in selected[:6])
        if len(selected) > 6:
            selected_times += ", ..."
        cv2.putText(canvas, f"EXACT FRAME  {position}/{total-1}  {position/fps:.3f}s  step={step}",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 230, 255), 2)
        cv2.putText(canvas, "Left/Right move | J step +10 | K step -10 | Left click add | Right click remove",
                    (20, 62), cv2.FONT_HERSHEY_SIMPLEX, .52, (20, 230, 255), 2)
        cv2.putText(canvas, f"Selected {len(selected)}: {selected_times} | Enter done | Esc cancel",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, .52, (20, 230, 255), 2)
        cv2.putText(canvas, status, (20, 118), cv2.FONT_HERSHEY_SIMPLEX, .48, (60, 220, 60), 1)
        badge_text = "CURRENT FRAME SELECTED" if is_selected else "CURRENT FRAME NOT SELECTED"
        badge_color = (40, 190, 40) if is_selected else (30, 30, 230)
        badge_x = max(10, canvas.shape[1] - 365)
        cv2.rectangle(canvas, (badge_x, 18), (canvas.shape[1] - 12, 55), (245, 245, 245), -1)
        cv2.putText(canvas, badge_text, (badge_x + 8, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    .58, badge_color, 2)
        cv2.imshow(title, canvas)
        if cv2.getTrackbarPos("frame", title) != position:
            scrub["programmatic"] = True
            cv2.setTrackbarPos("frame", title, position)
            scrub["programmatic"] = False
        key = cv2.waitKeyEx(30)
        direction = navigation_direction(key)
        if direction:
            position = calibration_move(position, step, direction, total)
            loaded_position = None
            status = f"Moved to exact frame {position}"
        elif key in (ord("j"), ord("J")):
            step = calibration_increase_step(step)
            status = f"Frame step is now {step}"
        elif key in (ord("k"), ord("K")):
            step = calibration_decrease_step(step)
            status = f"Frame step is now {step}"
        elif key == ord("s"):
            update_keyframe_selection(selected, position, add=position not in selected)
            status = f"Toggled exact frame {position}"
        elif key in (10, 13):
            if selected:
                break
            status = "Select at least one keyframe with lowercase s, or Esc to cancel"
        elif key == 27: selected = []; break
        elif cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            selected = []; break
    cap.release(); cv2.destroyWindow(title)
    return sorted(selected)


def read_frame(video_path: str | Path, frame_index: int) -> np.ndarray:
    """随机读取指定帧。"""
    cap = cv2.VideoCapture(str(video_path)); cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read(); cap.release()
    if not ok: raise RuntimeError(f"无法读取第 {frame_index} 帧")
    return frame
