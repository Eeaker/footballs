from __future__ import annotations

import csv
import json
from pathlib import Path

from analysis_lib.pipeline import run_analysis


def test_one_click_pipeline_writes_twenty_item_acceptance_sample(tmp_path: Path):
    tracking = tmp_path / "tracking"
    tracking.mkdir()
    mot_lines = []
    ball_rows = []
    # 26 stable possession runs => 25 A-to-B events, enough for the fixed 20-event sample.
    for run_index in range(26):
        identity = 1 if run_index % 2 == 0 else 2
        foot_x = 0.0 if identity == 1 else 10.0
        for offset in range(3):
            frame = run_index * 3 + offset
            mot_lines.append(f"{frame + 1},{identity},{foot_x - 1},-2,2,2,0.9,-1,-1,-1")
            ball_rows.append({"frame_proc": frame, "ball_x_px": foot_x + .25, "ball_y_px": 0, "observed": 1})
    (tracking / "tracking_mot.txt").write_text("\n".join(mot_lines) + "\n", encoding="utf-8")
    with (tracking / "ball_positions_observed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_proc", "ball_x_px", "ball_y_px", "observed"])
        writer.writeheader(); writer.writerows(ball_rows)
    (tracking / "tracking_run_metadata.json").write_text(json.dumps({
        "video": "/data/synthetic.mp4", "processed_fps": 10.0, "vid_stride": 1,
    }), encoding="utf-8")
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("""video:\n  path: /data/synthetic.mp4\ncalibration:\n  validated: true\n  mode: static\n  keyframes:\n    - frame_index: 0\n      homography:\n        - [1, 0, 0]\n        - [0, 1, 0]\n        - [0, 0, 1]\n""", encoding="utf-8")
    team_map = tmp_path / "teams.csv"
    team_map.write_text("global_id,team_id\n1,team_0\n2,team_0\n", encoding="utf-8")
    output = tmp_path / "output"
    report = run_analysis(
        tracking_dir=tracking, calibration=calibration, output=output, team_map_path=team_map,
    )
    assert report["status"] == "pending_human_review"
    assert report["counts"]["pass_events"] == 25
    with (output / "acceptance_sample_20.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 20
    assert (output / "pass_matrices.json").is_file()
    assert (output / "run_manifest.json").is_file()
