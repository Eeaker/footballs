from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _homography(image_points: list[list[float]], world_points: list[list[float]]) -> tuple[np.ndarray, dict]:
    if not 4 <= len(image_points) <= 8 or len(image_points) != len(world_points):
        raise ValueError("标定需要 4–8 对对应点")
    src = np.asarray(image_points, dtype=np.float64)
    dst = np.asarray(world_points, dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    if H is None:
        raise ValueError("标定点几何退化，无法求解")
    H = H / H[2, 2]
    proj = cv2.perspectiveTransform(src.astype(np.float32)[None], H)[0]
    errors = np.linalg.norm(proj - dst, axis=1)
    return H, {"max": float(errors.max()), "mean": float(errors.mean()), "rmse": float(np.sqrt(np.mean(errors ** 2)))}


def _project(H: np.ndarray, pts: list[list[float]]) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(pts, np.float32)[None], H)[0]


def build_reference_calibration(*, video: dict[str, Any], frame_index: int,
                                image_points: list[list[float]], world_points: list[list[float]],
                                validation_segments: list[dict[str, Any]], field_length_m: float,
                                field_width_m: float, tolerance_m: float, output: Path) -> dict[str, Any]:
    H, fit = _homography(image_points, world_points)
    if frame_index < 0 or frame_index >= int(video["frame_count"]):
        raise ValueError("参考帧超出视频范围")
    results = []
    for idx, segment in enumerate(validation_segments):
        p1 = segment.get("p1") or segment.get("image_points", [None, None])[0]
        p2 = segment.get("p2") or segment.get("image_points", [None, None])[1]
        length = float(segment.get("length_m") or segment.get("known_length_m"))
        if p1 is None or p2 is None or length <= 0:
            raise ValueError("验证线段必须提供两个图像点和已知米制长度")
        mapped = _project(H, [p1, p2])
        measured = float(np.linalg.norm(mapped[1] - mapped[0]))
        error = abs(measured - length)
        results.append({
            "name": segment.get("name") or f"validation_{idx + 1}",
            "known_length_m": length,
            "measured_length_m": measured,
            "absolute_error_m": error,
            "passed": error <= tolerance_m,
        })
    if not results:
        raise ValueError("正式标定至少需要 1 条未参与拟合的独立验证线段")
    passed = all(row["passed"] for row in results)
    payload = {
        "schema_version": 2,
        "camera_model": "single_fixed_homography",
        "warning": "Reference anchor only. Full-video metric output must use dynamic calibration.",
        "video": video["path"],
        "video_metadata": {
            "raw_fps": video["fps"], "raw_total_frames": video["frame_count"],
            "frame_width": video["width"], "frame_height": video["height"],
            "raw_frame_index": int(frame_index), "proc_fps": video["fps"],
            "proc_total_frames": video["frame_count"],
        },
        "vid_stride": 1,
        "calibration_proc_idx": int(frame_index),
        "valid_start_proc": int(frame_index), "valid_end_proc": int(frame_index),
        "image_points": image_points, "world_points_m": world_points,
        "H_image_to_pitch_m": H.tolist(),
        "field_bounds_m": {"x_min": 0.0, "x_max": float(field_length_m), "y_min": 0.0, "y_max": float(field_width_m), "margin_m": 0.5},
        "calibration_point_reprojection_error_m": fit,
        "validation_segments": [{
            "name": row.get("name") or f"validation_{i+1}",
            "image_points": [row.get("p1"), row.get("p2")],
            "known_length_m": float(row.get("length_m") or row.get("known_length_m")),
        } for i, row in enumerate(validation_segments)],
        "validation": {"tolerance_m": float(tolerance_m), "passed": passed, "results": results,
                       "max_absolute_error_m": max(r["absolute_error_m"] for r in results)},
        "provenance": {"source": "football_insight_manual_anchor_v2", "homography_solver": "OpenCV findHomography RANSAC"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def normalize_uploaded_dynamic(data: dict[str, Any], video: dict[str, Any], output: Path, min_coverage: float = 0.8) -> dict[str, Any]:
    if data.get("camera_model") != "dynamic_per_frame_homography":
        raise ValueError("只接受逐帧动态标定：camera_model 必须为 dynamic_per_frame_homography")
    meta = data.get("video_metadata") or {}
    checks = {
        "width": int(meta.get("frame_width", -1)) == int(video["width"]),
        "height": int(meta.get("frame_height", -1)) == int(video["height"]),
        "fps": abs(float(meta.get("proc_fps", -999)) - float(video["fps"])) <= 0.03,
        "frames": abs(int(meta.get("proc_total_frames", -999999)) - int(video["frame_count"])) <= 2,
        "stride": int(data.get("vid_stride", 1)) == 1,
    }
    if not all(checks.values()):
        failed = ", ".join(k for k, ok in checks.items() if not ok)
        raise ValueError(f"标定配置与当前视频不匹配：{failed}")
    if not bool(data.get("validation", {}).get("passed", False)):
        raise ValueError("标定配置尚未通过独立验证")
    frames = data.get("frames") or []
    accepted = sum(1 for row in frames if row.get("accepted") and row.get("H_image_to_pitch_m") is not None)
    ratio = accepted / len(frames) if frames else 1.0
    if frames and ratio < min_coverage:
        raise ValueError(f"动态标定有效帧比例过低：{accepted}/{len(frames)}")
    data = dict(data)
    data["video"] = video["path"]
    data["video_metadata"] = {**meta, "raw_fps": video["fps"], "proc_fps": video["fps"],
                              "raw_total_frames": video["frame_count"], "proc_total_frames": video["frame_count"],
                              "frame_width": video["width"], "frame_height": video["height"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    summary = _summarize_dynamic_data(data, fallback_total_frames=int(video["frame_count"]))
    summary.update({
        "passed": True,
        "accepted_frames": accepted or int(video["frame_count"]),
        "total_frames": len(frames) or int(video["frame_count"]),
        "accepted_ratio": ratio,
        "minimum_accepted_ratio": float(min_coverage),
    })
    return summary


def _summarize_dynamic_data(data: dict[str, Any], fallback_total_frames: int | None = None) -> dict[str, Any]:
    frames = data.get("frames") or []
    reg = data.get("dynamic_registration") or {}
    meta = data.get("video_metadata") or {}
    validation = data.get("validation") or {}
    bounds = data.get("field_bounds_m") or {}

    accepted = sum(1 for row in frames if row.get("accepted") and row.get("H_image_to_pitch_m") is not None)
    total = len(frames)
    if not total:
        accepted = int(reg.get("accepted_frame_count") or fallback_total_frames or 0)
        total = accepted + int(reg.get("rejected_frame_count") or 0)
        if not total:
            total = int(fallback_total_frames or 0)
            accepted = total

    anchor_indices = reg.get("anchor_proc_indices") or []
    reference_frame = reg.get("reference_proc_idx")
    if not anchor_indices and reference_frame is not None:
        anchor_indices = [reference_frame]
    anchors = data.get("anchors") or []
    anchor_count = len(anchors) or len(anchor_indices) or None

    error_values = []
    for row in validation.get("results") or []:
        try:
            error_values.append(float(row.get("absolute_error_m")))
        except (TypeError, ValueError):
            continue
    max_error = validation.get("max_absolute_error_m")
    if max_error is None and error_values:
        max_error = max(error_values)

    def number_or_none(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    x_min, x_max = number_or_none(bounds.get("x_min")), number_or_none(bounds.get("x_max"))
    y_min, y_max = number_or_none(bounds.get("y_min")), number_or_none(bounds.get("y_max"))
    return {
        "passed": bool(validation.get("passed", False)),
        "schema_version": data.get("schema_version"),
        "camera_model": data.get("camera_model"),
        "accepted_frames": accepted,
        "total_frames": total,
        "accepted_ratio": (accepted / total) if total else None,
        "fps": number_or_none(meta.get("proc_fps") or meta.get("raw_fps")),
        "frame_width": meta.get("frame_width"),
        "frame_height": meta.get("frame_height"),
        "vid_stride": data.get("vid_stride", 1),
        "valid_start_frame": data.get("valid_start_proc"),
        "valid_end_frame": data.get("valid_end_proc"),
        "field_bounds_m": bounds or None,
        "field_length_m": (x_max - x_min) if x_min is not None and x_max is not None else None,
        "field_width_m": (y_max - y_min) if y_min is not None and y_max is not None else None,
        "anchor_count": anchor_count,
        "anchor_proc_indices": anchor_indices,
        "reference_frame": reference_frame if reference_frame is not None else data.get("calibration_proc_idx"),
        "accepted_samples": reg.get("accepted_sample_count"),
        "sample_count": reg.get("sample_count"),
        "sample_step_frames": reg.get("sample_step_frames"),
        "max_interpolation_gap_frames": reg.get("max_interpolation_gap_frames"),
        "registration_method": reg.get("method"),
        "validation_tolerance_m": number_or_none(validation.get("tolerance_m")),
        "validation_max_error_m": number_or_none(max_error),
        "validation_segment_count": len(validation.get("results") or data.get("validation_segments") or []),
    }


def summarize_dynamic(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return _summarize_dynamic_data(data)


@lru_cache(maxsize=6)
def _cached_dynamic(path_text: str, modified_ns: int) -> dict[str, Any]:
    """Keep large per-frame calibration files in memory while the user scrubs."""
    del modified_ns
    return json.loads(Path(path_text).read_text(encoding="utf-8-sig"))


def _pitch_geometry(bounds: dict[str, Any]) -> list[dict[str, Any]]:
    x0, x1 = float(bounds.get("x_min", 0.0)), float(bounds.get("x_max", 45.0))
    y0, y1 = float(bounds.get("y_min", 0.0)), float(bounds.get("y_max", 25.0))
    length, width = x1 - x0, y1 - y0
    mid_x, mid_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    lines: list[dict[str, Any]] = [
        {"kind": "boundary", "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]},
        {"kind": "halfway", "points": [[mid_x, y0], [mid_x, y1]]},
    ]
    # A lightweight, field-size-aware tactical grid makes camera motion visible.
    for fraction in (0.25, 0.75):
        x = x0 + length * fraction
        lines.append({"kind": "guide", "points": [[x, y0], [x, y1]]})
    radius = min(4.5, width * 0.18, length * 0.12)
    circle = []
    for i in range(33):
        angle = 2.0 * np.pi * i / 32.0
        circle.append([mid_x + radius * np.cos(angle), mid_y + radius * np.sin(angle)])
    lines.append({"kind": "circle", "points": circle})
    return lines


def dynamic_frame_visualization(path: Path, frame_index: int) -> dict[str, Any]:
    """Return a browser-safe perspective pitch overlay for one calibrated frame."""
    resolved = path.resolve()
    data = _cached_dynamic(str(resolved), resolved.stat().st_mtime_ns)
    frames = data.get("frames") or []
    if not frames:
        raise ValueError("动态标定文件未包含逐帧矩阵")
    target = max(0, int(frame_index))
    row = frames[min(target, len(frames) - 1)]
    if int(row.get("proc_idx", -1)) != target:
        row = min(frames, key=lambda item: abs(int(item.get("proc_idx", 0)) - target))
    matrix = row.get("H_image_to_pitch_m")
    accepted = bool(row.get("accepted")) and matrix is not None
    bounds = data.get("field_bounds_m") or {"x_min": 0.0, "x_max": 45.0, "y_min": 0.0, "y_max": 25.0}
    projected: list[dict[str, Any]] = []
    inverse_list = None
    if matrix is not None:
        try:
            inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
            inverse /= inverse[2, 2]
            inverse_list = inverse.tolist()
            for line in _pitch_geometry(bounds):
                points = _project(inverse, line["points"])
                if np.isfinite(points).all():
                    projected.append({"kind": line["kind"], "points": np.round(points, 2).tolist()})
        except (ValueError, np.linalg.LinAlgError):
            accepted = False
    diagnostics = {}
    for key in ("matches", "inliers", "inlier_ratio", "residual_px", "source_anchor_proc_idx"):
        if row.get(key) is not None:
            diagnostics[key] = row[key]
    return {
        "requested_frame": target,
        "frame_index": int(row.get("proc_idx", target)),
        "accepted": accepted,
        "pitch_lines": projected,
        "H_pitch_m_to_image": inverse_list,
        "field_bounds_m": bounds,
        "source_width": (data.get("video_metadata") or {}).get("frame_width"),
        "source_height": (data.get("video_metadata") or {}).get("frame_height"),
        "diagnostics": diagnostics,
    }
