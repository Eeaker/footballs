"""Metric-coordinate trajectory smoothing and running metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np

from .homography import project_points
from .mot import MotDetection, group_by_identity_frame


@dataclass(frozen=True)
class TimeSeriesRow:
    global_id: int
    proc_idx: int
    time_sec: float
    foot_x_px: float
    foot_y_px: float
    x_m_raw: float
    y_m_raw: float
    x_m_smooth: float
    y_m_smooth: float
    step_distance_m: float | None
    speed_mps: float | None
    segment_id: int


def median_smooth(values: Iterable[float], window: int = 11) -> np.ndarray:
    data = np.asarray(list(values), dtype=np.float64)
    if window < 1 or window % 2 == 0:
        raise ValueError("median window must be a positive odd integer")
    if len(data) == 0:
        return data
    radius = window // 2
    padded = np.pad(data, (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(windows, axis=1)


def _inside_bounds(point: np.ndarray, bounds: dict | None) -> bool:
    if bounds is None:
        return True
    margin = float(bounds.get("margin_m", 0.0))
    return (
        float(bounds["x_min"]) - margin <= point[0] <= float(bounds["x_max"]) + margin
        and float(bounds["y_min"]) - margin <= point[1] <= float(bounds["y_max"]) + margin
    )


def _consecutive_runs(samples: list[tuple[int, MotDetection, np.ndarray]]) -> list[list]:
    runs: list[list] = []
    current: list = []
    for sample in samples:
        if current and sample[0] != current[-1][0] + 1:
            runs.append(current)
            current = []
        current.append(sample)
    if current:
        runs.append(current)
    return runs


def calculate_running_metrics(
    records: list[MotDetection],
    homography: np.ndarray | Mapping[int, np.ndarray],
    proc_fps: float,
    valid_start_proc: int,
    valid_end_proc: int,
    field_bounds: dict | None = None,
    median_window: int = 11,
    high_speed_threshold_mps: float = 4.5,
) -> tuple[list[dict], list[TimeSeriesRow], dict]:
    if proc_fps <= 0:
        raise ValueError("proc_fps must be positive")
    if valid_start_proc < 0 or valid_end_proc < valid_start_proc:
        raise ValueError("invalid processed-frame interval")
    if median_window < 1 or median_window % 2 == 0:
        raise ValueError("median_window must be a positive odd integer")

    grouped = group_by_identity_frame(
        [r for r in records if valid_start_proc <= r.proc_idx <= valid_end_proc]
    )
    interval_frames = valid_end_proc - valid_start_proc + 1
    summaries: list[dict] = []
    all_rows: list[TimeSeriesRow] = []
    identity_quality: list[dict] = []

    for gid, frame_map in sorted(grouped.items()):
        collision_frames = sorted(f for f, rows in frame_map.items() if len(rows) > 1)
        unique = [rows[0] for f, rows in sorted(frame_map.items()) if len(rows) == 1]
        valid_samples: list[tuple[int, MotDetection, np.ndarray]] = []
        out_of_bounds = 0
        missing_calibration = 0
        for record in unique:
            if isinstance(homography, Mapping):
                frame_h = homography.get(record.proc_idx)
                if frame_h is None:
                    missing_calibration += 1
                    continue
            else:
                frame_h = homography
            point = project_points([(record.foot_x, record.foot_y)], frame_h)[0]
            if _inside_bounds(point, field_bounds):
                valid_samples.append((record.proc_idx, record, point))
            else:
                out_of_bounds += 1

        runs = _consecutive_runs(valid_samples)
        eligible_runs = [run for run in runs if len(run) >= median_window]
        short_segment_frames = sum(len(run) for run in runs if len(run) < median_window)
        speeds: list[float] = []
        distances: list[float] = []
        segment_counter = 0

        for run in eligible_runs:
            segment_counter += 1
            raw_xy = np.asarray([sample[2] for sample in run], dtype=np.float64)
            smooth_x = median_smooth(raw_xy[:, 0], median_window)
            smooth_y = median_smooth(raw_xy[:, 1], median_window)
            smooth_xy = np.column_stack([smooth_x, smooth_y])
            steps = np.linalg.norm(np.diff(smooth_xy, axis=0), axis=1)
            run_speeds = steps * proc_fps
            distances.extend(float(value) for value in steps)
            speeds.extend(float(value) for value in run_speeds)

            for index, (proc_idx, record, raw_point) in enumerate(run):
                step = None if index == 0 else float(steps[index - 1])
                speed = None if index == 0 else float(run_speeds[index - 1])
                all_rows.append(
                    TimeSeriesRow(
                        global_id=gid,
                        proc_idx=proc_idx,
                        time_sec=proc_idx / proc_fps,
                        foot_x_px=record.foot_x,
                        foot_y_px=record.foot_y,
                        x_m_raw=float(raw_point[0]),
                        y_m_raw=float(raw_point[1]),
                        x_m_smooth=float(smooth_xy[index, 0]),
                        y_m_smooth=float(smooth_xy[index, 1]),
                        step_distance_m=step,
                        speed_mps=speed,
                        segment_id=segment_counter,
                    )
                )

        distance_array = np.asarray(distances, dtype=np.float64)
        speed_array = np.asarray(speeds, dtype=np.float64)
        total_distance = float(distance_array.sum()) if len(distance_array) else 0.0
        high_speed_distance = float(
            distance_array[speed_array > high_speed_threshold_mps].sum()
        ) if len(speed_array) else 0.0
        peak_speed = float(np.percentile(speed_array, 95)) if len(speed_array) else 0.0
        observed_frames = len(frame_map)
        flags = []
        if collision_frames:
            flags.append("identity_collision")
        if out_of_bounds:
            flags.append("out_of_bounds")
        if missing_calibration:
            flags.append("missing_frame_calibration")
        if short_segment_frames:
            flags.append("short_segments_excluded")
        if observed_frames / interval_frames < 0.5:
            flags.append("low_visible_coverage")
        if not speeds:
            flags.append("no_valid_steps")

        summaries.append(
            {
                "global_id": gid,
                "total_distance_m": round(total_distance, 3),
                "high_speed_distance_m": round(high_speed_distance, 3),
                "peak_speed_mps_p95": round(peak_speed, 3),
                "valid_steps": len(speeds),
                "valid_duration_sec": round(len(speeds) / proc_fps, 3),
                "eligible_frames": sum(len(run) for run in eligible_runs),
                "observed_unique_frames": observed_frames,
                "interval_coverage": round(observed_frames / interval_frames, 6),
                "continuous_segments": len(eligible_runs),
                "collision_frames": len(collision_frames),
                "out_of_bounds_frames": out_of_bounds,
                "missing_calibration_frames": missing_calibration,
                "short_segment_frames": short_segment_frames,
                "quality_flags": ";".join(flags),
            }
        )
        indices = sorted(frame_map)
        gaps = [b - a for a, b in zip(indices, indices[1:])]
        identity_quality.append(
            {
                "global_id": gid,
                "collision_proc_indices": collision_frames,
                "out_of_bounds_frames": out_of_bounds,
                "missing_calibration_frames": missing_calibration,
                "short_segment_frames": short_segment_frames,
                "max_gap_frames": max(gaps, default=0),
                "quality_flags": flags,
            }
        )

    quality = {
        "valid_interval": {"start_proc": valid_start_proc, "end_proc": valid_end_proc},
        "proc_fps": proc_fps,
        "median_window_frames": median_window,
        "high_speed_threshold_mps": high_speed_threshold_mps,
        "gap_policy": "only delta_frame=1 contributes; gaps split segments",
        "duplicate_policy": "duplicate (global_id, proc_idx) frames are excluded",
        "calibration_policy": (
            "per-frame H; missing/rejected H splits trajectory"
            if isinstance(homography, Mapping)
            else "single fixed H"
        ),
        "identities": identity_quality,
    }
    return summaries, sorted(all_rows, key=lambda r: (r.proc_idx, r.global_id)), quality


def timeseries_as_dicts(rows: list[TimeSeriesRow]) -> list[dict]:
    return [asdict(row) for row in rows]
