"""Render a standalone metric pitch video with possession and pass-network state.

The screen-view y orientation, ID colour helper, CSV grouping pattern, centre
line and 3 m centre-circle convention are migrated from
``football_metric_running/src/running_metrics_v1/render_demo.py``.
This adapter intentionally renders a new canvas instead of modifying source
video frames, then adds match analysis possession/transition/pass overlays.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import cv2
import numpy as np


TEAM_COLORS = {
    "team_0": (70, 215, 255),
    "team_1": (255, 190, 70),
    "team_2": (210, 100, 255),
    "unassigned": (180, 180, 180),
}


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class PitchMapper:
    def __init__(self, width: int, height: int, bounds: dict, x0: int = 0, y0: int = 0):
        self.width, self.height, self.bounds = width, height, bounds
        self.x0, self.y0 = x0, y0

    def to_px(self, x_m: float, y_m: float) -> tuple[int, int]:
        xmin, xmax = float(self.bounds["x_min"]), float(self.bounds["x_max"])
        ymin, ymax = float(self.bounds["y_min"]), float(self.bounds["y_max"])
        px = self.x0 + int(round((x_m - xmin) / max(xmax - xmin, 1e-9) * self.width))
        py = self.y0 + int(round((y_m - ymin) / max(ymax - ymin, 1e-9) * self.height))
        return px, py


def active_event_at_frame(event: dict, frame_proc: int) -> bool:
    return int(event["release_frame_proc"]) <= frame_proc <= int(event["receive_confirmed_frame_proc"])


def _group_int(rows: list[dict], key: str) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        result[int(row[key])].append(row)
    return dict(result)


def _draw_pitch(canvas: np.ndarray, mapper: PitchMapper) -> None:
    x0, y0 = mapper.x0, mapper.y0
    x1, y1 = x0 + mapper.width, y0 + mapper.height
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (35, 118, 48), -1)
    stripe_w = max(1, mapper.width // 10)
    for index in range(0, 10, 2):
        cv2.rectangle(canvas, (x0 + index * stripe_w, y0),
                      (min(x1, x0 + (index + 1) * stripe_w), y1), (39, 126, 52), -1)
    line = (238, 238, 238)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), line, 3)
    bounds = mapper.bounds
    centre_x = (float(bounds["x_min"]) + float(bounds["x_max"])) / 2
    centre_y = (float(bounds["y_min"]) + float(bounds["y_max"])) / 2
    cv2.line(canvas, mapper.to_px(centre_x, float(bounds["y_min"])),
             mapper.to_px(centre_x, float(bounds["y_max"])), line, 2)
    centre = mapper.to_px(centre_x, centre_y)
    radius = max(1, int(round(3.0 / (float(bounds["x_max"]) - float(bounds["x_min"])) * mapper.width)))
    cv2.circle(canvas, centre, radius, line, 2)
    cv2.circle(canvas, centre, 4, line, -1)


def _put(canvas: np.ndarray, text: str, xy: tuple[int, int], scale: float = .6,
         color: tuple[int, int, int] = (245, 245, 245), thickness: int = 1) -> None:
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_pitch_video(
    *, calibration_path: str | Path, timeseries_path: str | Path,
    possession_path: str | Path, transitions_path: str | Path,
    passes_path: str | Path, team_map_path: str | Path,
    output_path: str | Path, width: int = 1280, height: int = 720,
    start_proc: int | None = None, end_proc: int | None = None,
) -> dict:
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite pitch video: {output_path}")
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    bounds = calibration["field_bounds_m"]
    fps = float(calibration["video_metadata"]["proc_fps"])
    start = int(calibration["valid_start_proc"])
    end = int(calibration["valid_end_proc"])
    if start_proc is not None:
        start = max(start, start_proc)
    if end_proc is not None:
        end = min(end, end_proc)
    timeseries = read_csv(timeseries_path)
    if timeseries:
        end = min(end, max(int(row["proc_idx"]) for row in timeseries))
    if end < start:
        raise ValueError("requested render frame interval is empty")
    by_frame = _group_int(timeseries, "proc_idx")
    possession_by_frame = {
        int(row["frame_proc"]): row for row in read_csv(possession_path)
    }
    transitions = read_csv(transitions_path)
    passes = read_csv(passes_path)
    transitions_by_release = _group_int(transitions, "release_frame_proc")
    passes_by_confirm = _group_int(passes, "receive_confirmed_frame_proc")
    team_map = {int(row["global_id"]): row["team_id"] for row in read_csv(team_map_path)}

    margin, sidebar = 38, 220
    usable_w = width - sidebar - margin * 3
    usable_h = height - margin * 2
    field_ratio = ((float(bounds["x_max"]) - float(bounds["x_min"])) /
                   (float(bounds["y_max"]) - float(bounds["y_min"])))
    pitch_w = min(usable_w, int(round(usable_h * field_ratio)))
    pitch_h = min(usable_h, int(round(pitch_w / field_ratio)))
    mapper = PitchMapper(pitch_w, pitch_h, bounds, margin, (height - pitch_h) // 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create pitch video: {output_path}")

    cumulative_distance = Counter()
    network_counts = Counter()
    network_latest: dict[tuple[str, int, int], dict] = {}
    transition_total = pass_total = 0
    try:
        for frame_proc in range(start, end + 1):
            canvas = np.full((height, width, 3), (20, 25, 28), dtype=np.uint8)
            _draw_pitch(canvas, mapper)
            for row in passes_by_confirm.get(frame_proc, []):
                key = (row["team_id"], int(row["from_global_id"]), int(row["to_global_id"]))
                network_counts[key] += 1
                network_latest[key] = row
                pass_total += 1
            transition_total += len(transitions_by_release.get(frame_proc, []))

            for key, count in network_counts.items():
                row = network_latest[key]
                start_px = mapper.to_px(float(row["start_x_m"]), float(row["start_y_m"]))
                end_px = mapper.to_px(float(row["end_x_m"]), float(row["end_y_m"]))
                color = TEAM_COLORS.get(key[0], TEAM_COLORS["unassigned"])
                cv2.arrowedLine(canvas, start_px, end_px, color, min(6, 1 + count), cv2.LINE_AA, tipLength=.12)

            active_transitions = [row for row in transitions if active_event_at_frame(row, frame_proc)]
            active_passes = [row for row in passes if active_event_at_frame(row, frame_proc)]
            for row in active_transitions:
                cv2.arrowedLine(
                    canvas,
                    mapper.to_px(float(row["start_x_m"]), float(row["start_y_m"])),
                    mapper.to_px(float(row["end_x_m"]), float(row["end_y_m"])),
                    (0, 145, 255), 2, cv2.LINE_AA, tipLength=.16,
                )
            for row in active_passes:
                cv2.arrowedLine(
                    canvas,
                    mapper.to_px(float(row["start_x_m"]), float(row["start_y_m"])),
                    mapper.to_px(float(row["end_x_m"]), float(row["end_y_m"])),
                    (80, 255, 120), 4, cv2.LINE_AA, tipLength=.18,
                )

            holder = possession_by_frame.get(frame_proc)
            for row in by_frame.get(frame_proc, []):
                gid = int(row["global_id"])
                step = row.get("step_distance_m", "")
                if step not in (None, ""):
                    cumulative_distance[gid] += float(step)
                point = mapper.to_px(float(row["x_m_smooth"]), float(row["y_m_smooth"]))
                team = team_map.get(gid, "unassigned")
                color = TEAM_COLORS.get(team, TEAM_COLORS["unassigned"])
                if holder and int(holder["global_id"]) == gid:
                    cv2.circle(canvas, point, 14, (0, 255, 255), 3)
                cv2.circle(canvas, point, 8, color, -1)
                cv2.circle(canvas, point, 9, (255, 255, 255), 1)
                speed = row.get("speed_mps", "")
                label = f"{gid}" if speed in (None, "") else f"{gid} {float(speed):.1f}m/s"
                _put(canvas, label, (point[0] + 10, point[1] - 8), .44, (255, 255, 255), 1)

            if holder:
                ball = mapper.to_px(float(holder["ball_x_m"]), float(holder["ball_y_m"]))
                cv2.circle(canvas, ball, 5, (255, 255, 255), -1)
                cv2.circle(canvas, ball, 7, (0, 0, 0), 1)

            sx = width - sidebar
            _put(canvas, "METRIC MATCH PLANE", (sx, 48), .63, (255, 255, 255), 2)
            _put(canvas, f"time  {frame_proc / fps:8.2f}s", (sx, 82), .55)
            holder_text = "none" if not holder else f"ID {holder['global_id']} / {holder['team_id']}"
            _put(canvas, f"possession: {holder_text}", (sx, 116), .48, (0, 255, 255))
            _put(canvas, f"A->B changes: {transition_total}", (sx, 154), .5, (0, 165, 255))
            _put(canvas, f"active passes: {pass_total}", (sx, 184), .5, (80, 255, 120))
            _put(canvas, "RUNNING DISTANCE", (sx, 232), .52, (230, 230, 230), 1)
            for index, (gid, distance) in enumerate(cumulative_distance.most_common(10)):
                _put(canvas, f"ID {gid:>2}  {distance:7.1f} m", (sx, 262 + index * 29), .48,
                     TEAM_COLORS.get(team_map.get(gid, "unassigned"), TEAM_COLORS["unassigned"]), 1)
            _put(canvas, "yellow ring = possession", (sx, height - 55), .42, (0, 255, 255), 1)
            _put(canvas, "green = pass / orange = change", (sx, height - 28), .39, (220, 220, 220), 1)
            writer.write(canvas)
    finally:
        writer.release()
    return {
        "output": str(output_path.resolve()), "frames": end - start + 1,
        "fps": fps, "width": width, "height": height,
        "possession_transition_rows": len(transitions), "active_pass_rows": len(passes),
        "source_renderer": "football_metric_running/render_demo.py adapted to standalone canvas",
    }
