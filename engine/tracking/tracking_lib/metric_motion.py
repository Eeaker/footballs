from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .homography import HomographyProvider, project_footpoint


def export_metric_motion(mot_path: str | Path, output_dir: str | Path, calibration: dict,
                         processed_fps: float, vid_stride: int = 1) -> dict:
    """仅在标定独立验证通过后导出米制位置和可见段跑动距离。"""
    if not calibration.get("enabled") or not calibration.get("validated"):
        return {"enabled": False, "reason": "calibration_not_validated"}
    provider = HomographyProvider(calibration); rows = []
    for line in Path(mot_path).read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        frame, identity = int(float(parts[0])), int(float(parts[1]))
        x, y, width, height = map(float, parts[2:6])
        homography = provider.at(max(0, frame - 1) * max(1, vid_stride))
        if homography is None:
            continue
        mx, my = project_footpoint(homography, x, y, width, height)
        if np.isfinite(mx) and np.isfinite(my):
            rows.append((frame, identity, mx, my))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[1]].append(row)
    players = []
    for identity, values in sorted(grouped.items()):
        values.sort(); distance = 0.0
        for previous, current in zip(values, values[1:]):
            if current[0] - previous[0] <= 2:
                distance += float(np.hypot(current[2] - previous[2], current[3] - previous[3]))
        players.append({"global_id": identity, "visible_samples": len(values),
                        "distance_m": round(distance, 3),
                        "visible_seconds": round(len(values) / max(processed_fps, 1e-9), 3)})
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    with (output / "metric_positions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle); writer.writerow(["frame_proc", "global_id", "x_m", "y_m"])
        writer.writerows((f, i, round(x, 3), round(y, 3)) for f, i, x, y in rows)
    report = {"enabled": True, "position_rows": len(rows), "players": players,
              "warning": "距离只累计连续可见帧，受ID连续性与标定质量共同影响。"}
    (output / "metric_motion_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
