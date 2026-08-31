from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from generate_player_card import generate_from_mot, resolve_calibration


ROOT = Path(__file__).resolve().parents[1]
RUNNING_SRC = ROOT.parent / "football_metric_running" / "src"


def _write_video(path: Path, frames: int = 20, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((48, 64, 3), (index * 7) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_calibration(path: Path) -> None:
    path.write_text(json.dumps({
        "camera_model": "single_fixed_homography",
        "H_image_to_pitch_m": [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]],
        "valid_start_proc": 0,
        "valid_end_proc": 19,
        "field_bounds_m": {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10, "margin_m": 0},
        "validation": {"passed": True},
        "video_metadata": {
            "raw_fps": 10.0, "proc_fps": 10.0,
            "raw_total_frames": 20, "proc_total_frames": 20,
            "frame_width": 64, "frame_height": 48,
        },
    }), encoding="utf-8")


def test_four_input_entry_generates_complete_package_and_crops_event(tmp_path):
    video = tmp_path / "match.mp4"
    mot = tmp_path / "tracking_mot.txt"
    numbers = tmp_path / "clip_eligibility.json"
    events = tmp_path / "events.json"
    calibration = tmp_path / "dynamic_calibration.json"
    output = tmp_path / "player_cards"
    _write_video(video)
    _write_calibration(calibration)
    mot.write_text("".join(
        f"{frame},1,{10 + frame}.0,20.0,4.0,10.0,0.9,-1,-1,-1\n"
        for frame in range(1, 21)
    ), encoding="utf-8")
    numbers.write_text(json.dumps({
        "eligible_confirmed": [{"global_id": 1, "team": "white", "final_number": 26, "confidence": 0.9}],
        "excluded_conflict": [], "excluded_mismatch": [], "excluded_unreadable": [],
    }), encoding="utf-8")
    events.write_text(json.dumps({"events": [{
        "event_id": "ev001", "start_time": 0.2, "end_time": 0.8,
        "primary_global_id": 1, "event_type": "pass", "confidence": 0.8,
    }]}), encoding="utf-8")

    manifest = generate_from_mot(
        video=video, mot=mot, numbers=numbers, events=events, output=output,
        calibration=calibration, running_src=RUNNING_SRC,
    )

    assert manifest["players"] == ["white_26"]
    assert (output / "player_running_timeseries.csv").is_file()
    assert (output / "running_quality_report.json").is_file()
    assert (output / "white_26" / "running.json").is_file()
    assert (output / "white_26" / "heatmap.png").is_file()
    assert (output / "white_26" / "assessment_report_input.json").is_file()
    assert (output / "white_26" / "highlights" / "white_26_ev001.mp4").is_file()
    saved_manifest = json.loads((output / "package_manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["inputs"]["mot"]["sha256"]


def test_calibration_auto_discovery_prefers_matching_dynamic_file(tmp_path, monkeypatch):
    video = tmp_path / "match.mp4"
    mot = tmp_path / "tracking_mot.txt"
    calibration = tmp_path / "dynamic_calibration.json"
    _write_video(video)
    mot.write_text("1,1,1,1,1,1,0.9,-1,-1,-1\n", encoding="utf-8")
    _write_calibration(calibration)
    data = json.loads(calibration.read_text(encoding="utf-8"))
    data["camera_model"] = "dynamic_per_frame_homography"
    calibration.write_text(json.dumps(data), encoding="utf-8")

    assert resolve_calibration(video, mot) == calibration.resolve()
