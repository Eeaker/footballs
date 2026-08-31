from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class MOTBox:
    frame_proc: int
    global_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float

    @property
    def footpoint_px(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height


@dataclass(frozen=True)
class BallPoint:
    frame_proc: int
    x: float
    y: float
    source: str


def read_mot(path: str | Path) -> tuple[dict[int, list[MOTBox]], list[MOTBox]]:
    by_frame: dict[int, list[MOTBox]] = defaultdict(list)
    rows: list[MOTBox] = []
    seen: set[tuple[int, int]] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 7:
            raise ValueError(f"MOT 第 {line_number} 行少于7列")
        # MOT 是 1-based；tracking ball_positions_observed.csv 是 0-based。
        row = MOTBox(
            int(float(parts[0])) - 1, int(float(parts[1])),
            *map(float, parts[2:7]),
        )
        key = (row.frame_proc, row.global_id)
        if key in seen:
            raise ValueError(f"同帧 global_id 不唯一: frame_proc={key[0]}, global_id={key[1]}")
        seen.add(key)
        rows.append(row)
        by_frame[row.frame_proc].append(row)
    if not rows:
        raise ValueError(f"MOT 为空: {path}")
    return dict(by_frame), rows


def read_ball(path: str | Path, max_gap_frames: int = 0) -> dict[int, BallPoint]:
    observed: dict[int, BallPoint] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("observed", "1")).strip().lower() not in {"1", "true", "yes"}:
                continue
            frame = int(row["frame_proc"])
            observed[frame] = BallPoint(frame, float(row["ball_x_px"]), float(row["ball_y_px"]), "observed")
    if not observed:
        raise ValueError(f"球轨迹为空: {path}")
    if max_gap_frames <= 0:
        return observed
    result = dict(observed)
    frames = sorted(observed)
    for left, right in zip(frames, frames[1:]):
        gap = right - left - 1
        if gap <= 0 or gap > max_gap_frames:
            continue
        a, b = observed[left], observed[right]
        for frame in range(left + 1, right):
            ratio = (frame - left) / (right - left)
            result[frame] = BallPoint(
                frame, a.x * (1 - ratio) + b.x * ratio, a.y * (1 - ratio) + b.y * ratio,
                "interpolated",
            )
    return result


def read_metadata(path: str | Path | None) -> dict:
    if path is None or not Path(path).is_file():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_team_map(path: str | Path) -> dict[int, str]:
    result: dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[int(row["global_id"])] = str(row["team_id"])
    if not result:
        raise ValueError(f"队伍映射为空: {path}")
    return result


def write_csv(path: str | Path, rows: list[dict], columns: list[str]) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
