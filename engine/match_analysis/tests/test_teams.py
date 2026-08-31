from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from analysis_lib.io import MOTBox
from analysis_lib.teams import assign_teams_kmeans


def test_track_level_kmeans_separates_two_jersey_colors(tmp_path: Path):
    video = tmp_path / "teams.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (120, 120))
    assert writer.isOpened()
    boxes = {
        1: (10, (0, 0, 255)), 2: (35, (0, 0, 255)),
        3: (65, (255, 0, 0)), 4: (90, (255, 0, 0)),
    }
    mot_rows = []
    for frame_index in range(8):
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        for identity, (x, color) in boxes.items():
            cv2.rectangle(frame, (x, 20), (x + 18, 80), color, -1)
            mot_rows.append(MOTBox(frame_index, identity, x, 20, 18, 60, .9))
        writer.write(frame)
    writer.release()
    team_map, diagnostics = assign_teams_kmeans(video, mot_rows, 2, samples_per_id=4)
    assert team_map[1] == team_map[2]
    assert team_map[3] == team_map[4]
    assert team_map[1] != team_map[3]
    assert all(row["samples"] == 4 for row in diagnostics)
