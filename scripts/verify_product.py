from __future__ import annotations

import io
import json

import cv2
import numpy as np
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.app import app
from app.services.storage import load_project, save_project, project_dir


def require(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)



def _make_verification_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (320, 180))
    if not writer.isOpened():
        raise RuntimeError("unable to create product-verification video")
    for frame_index in range(24):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (40 + frame_index * 8, 90), 8, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

def main() -> None:
    client = TestClient(app)
    checks: list[str] = []

    r = client.get("/api/health")
    require(r.status_code == 200 and r.json().get("version") == "2.3.3", "health failed")
    checks.append("health")

    r = client.get("/api/system/status")
    require(r.status_code == 200 and "readiness" in r.json(), "system readiness missing")
    checks.append("system readiness")

    # Model management must reject obviously invalid/small uploads and must not install a fake weight.
    r = client.post("/api/system/model", files={"model": ("bad.pt", io.BytesIO(b"x" * 1024), "application/octet-stream")})
    require(r.status_code == 400, "invalid model upload was not rejected")
    checks.append("model management validation")

    require(client.get("/static/user_guide.html").status_code == 200, "user guide missing")
    index_html = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    for marker in ["settingsImportInput", "exportSettingsBtn", "downloadCalibrationBtn", "uploadedCalibrationSummary", "calibFrameBadge", "overviewStory"]:
        require(marker in index_html, f"formal UI control missing: {marker}")
    checks.append("user guide + formal configuration UI")

    demo_id = "demo-reference-match"
    for endpoint in ["overview", "pitch", "replay", "events", "highlights", "players", "quality", "files"]:
        r = client.get(f"/api/projects/{demo_id}/{endpoint}")
        require(r.status_code == 200, f"demo {endpoint} failed: {r.text[:200]}")
    require(client.get(f"/api/projects/{demo_id}/report").status_code == 200, "demo report failed")
    checks.append("demo read-only result center")

    # Create a formal project using the public API.
    r = client.post("/api/projects", data={"name": "V2-verification", "home_team": "蓝队", "away_team": "橙队", "age_group": "U12"})
    require(r.status_code == 200, r.text)
    pid = r.json()["id"]
    try:
        video = project_dir(pid) / "verification_input.mp4"
        _make_verification_video(video)
        with video.open("rb") as f:
            r = client.post(f"/api/projects/{pid}/video", files={"video": (video.name, f, "video/mp4")})
        require(r.status_code == 200, f"video upload failed: {r.text}")
        checks.append("formal video upload/probe")

        # Portable parameter templates: weights path never travels with the template and vid_stride stays 1.
        r = client.get(f"/api/projects/{pid}/settings/export")
        require(r.status_code == 200 and "attachment" in r.headers.get("content-disposition", ""), "settings export failed")
        exported = json.loads(r.content.decode("utf-8"))
        require("weights_path" not in exported.get("settings", {}), "portable settings leaked weights_path")
        imported = {"settings": {**exported["settings"], "confidence": 0.31, "imgsz": 960, "vid_stride": 9, "weights_path": "C:/wrong/model.pt"}}
        r = client.post(f"/api/projects/{pid}/settings/import", files={"config": ("parameters.json", io.BytesIO(json.dumps(imported).encode("utf-8")), "application/json")})
        require(r.status_code == 200, f"settings import failed: {r.text}")
        current = client.get(f"/api/projects/{pid}").json()["settings"]
        require(abs(float(current["confidence"]) - 0.31) < 1e-9 and int(current["imgsz"]) == 960, "settings import values not applied")
        require(int(current["vid_stride"]) == 1 and "wrong/model.pt" not in str(current["weights_path"]), "portable settings bypassed safety rules")
        checks.append("portable parameter import/export")

        # Existing same-video dynamic calibration upload is a supported formal path.
        # Build a metadata-matching synthetic config for API/schema smoke testing only;
        # it is never used to claim metric accuracy.
        import tempfile
        meta = client.get(f"/api/projects/{pid}").json()["video"]
        cfg = {
            "schema_version": 2, "camera_model": "dynamic_per_frame_homography", "vid_stride": 1,
            "valid_start_proc": 0, "valid_end_proc": meta["frame_count"] - 1, "calibration_proc_idx": 12,
            "image_points": [[100,100],[200,100],[200,200],[100,200]], "world_points_m": [[0,0],[45,0],[45,25],[0,25]],
            "validation_segments": [{"name": "touchline", "image_points": [[100,100],[200,100]], "known_length_m": 45}],
            "video_metadata": {"raw_fps": meta["fps"], "proc_fps": meta["fps"], "raw_total_frames": meta["frame_count"], "proc_total_frames": meta["frame_count"], "frame_width": meta["width"], "frame_height": meta["height"]},
            "field_bounds_m": {"x_min": 0, "x_max": 45, "y_min": 0, "y_max": 25},
            "validation": {"passed": True, "tolerance_m": 0.5, "results": [{"absolute_error_m": 0.2, "passed": True}], "note": "product verifier schema smoke only"},
            "dynamic_registration": {"method": "verification", "reference_proc_idx": 12, "sample_step_frames": 5, "max_interpolation_gap_frames": 30, "accepted_sample_count": 10, "sample_count": 10},
            "frames": [{"proc_idx": i, "accepted": True, "H_image_to_pitch_m": [[1,0,0],[0,1,0],[0,0,1]]} for i in range(meta["frame_count"])],
        }
        tmp = project_dir(pid) / "calibration" / "verification_dynamic.json"
        tmp.write_text(json.dumps(cfg), encoding="utf-8")
        with tmp.open("rb") as f:
            r = client.post(f"/api/projects/{pid}/calibration/upload", files={"config": (tmp.name, f, "application/json")})
        require(r.status_code == 200, f"calibration upload failed: {r.text}")
        uploaded = r.json()
        cal_summary = uploaded.get("validation") or {}
        require(uploaded.get("source_filename") == tmp.name, "uploaded calibration filename was not persisted")
        require(cal_summary.get("frame_width") == meta["width"] and cal_summary.get("frame_height") == meta["height"], "calibration video summary missing")
        require(cal_summary.get("valid_end_frame") == meta["frame_count"] - 1 and cal_summary.get("field_length_m") == 45.0, "calibration range summary missing")
        require(cal_summary.get("validation_max_error_m") == 0.2 and cal_summary.get("sample_step_frames") == 5, "calibration validation detail missing")
        visual = client.get(f"/api/projects/{pid}/calibration/visualization")
        require(visual.status_code == 200, f"calibration visualization failed: {visual.text}")
        require(visual.json().get("frame_index") == 12 and len(visual.json().get("image_points") or []) == 4, "calibration frame/points missing")
        require(len(visual.json().get("validation_segments") or []) == 1, "calibration validation overlay missing")
        r = client.get(f"/api/projects/{pid}/calibration/download")
        require(r.status_code == 200 and len(r.content) > 1000, "calibration download failed")
        checks.append("formal dynamic calibration upload/summary/download")

        # Seed only result fixtures so the non-GPU product layer can be verified end-to-end.
        # This does NOT claim a new inference run; it validates result ingestion/review/report/export.
        out = project_dir(pid) / "outputs"
        shutil.copytree(ROOT / "demo_data" / "reference_match" / "match_analysis", out / "match_analysis", dirs_exist_ok=True)
        shutil.copytree(ROOT / "demo_data" / "reference_match" / "number_ocr", out / "number_ocr", dirs_exist_ok=True)
        shutil.copytree(ROOT / "demo_data" / "reference_match" / "player_cards", out / "player_cards", dirs_exist_ok=True)
        highlights = ROOT / "demo_data" / "reference_match" / "highlights"
        if highlights.is_dir():
            shutil.copytree(highlights, out / "highlights", dirs_exist_ok=True)
        p = load_project(pid)
        p["pipeline"]["state"] = "complete"
        p["pipeline"]["progress"] = 100
        p["status"] = "complete"
        for step in p["pipeline"]["steps"]:
            step.update(state="complete", progress=100, message="verification fixture")
        save_project(p)

        for endpoint in ["overview", "pitch", "events", "highlights", "players", "quality", "files"]:
            r = client.get(f"/api/projects/{pid}/{endpoint}")
            require(r.status_code == 200, f"formal result {endpoint} failed: {r.text[:200]}")
        checks.append("formal result ingestion")

        # Pass review loop.
        r = client.get(f"/api/projects/{pid}/reviews/passes")
        require(r.status_code == 200 and r.json()["total"] > 0, "pass review source missing")
        first_key = r.json()["rows"][0]["key"]
        r = client.put(f"/api/projects/{pid}/reviews/passes/{first_key}", json={"human_is_pass": True, "outcome": "", "note": "verification"})
        require(r.status_code == 200 and r.json()["labeled"] == 1, "pass review save failed")
        checks.append("pass human review")

        # Identity mapping loop.
        r = client.get(f"/api/projects/{pid}/reviews/identities")
        require(r.status_code == 200 and r.json()["candidate_global_ids"], "identity candidates missing")
        gid = r.json()["candidate_global_ids"][0]
        r = client.put(f"/api/projects/{pid}/reviews/identities/{gid}", json={"name": "测试球员", "jersey_number": "9", "team_id": "team_0", "note": "verification"})
        require(r.status_code == 200 and r.json()["confirmed"] >= 1, "identity mapping save failed")
        ps = client.get(f"/api/projects/{pid}/players").json()
        require(any(x.get("player_id") == "测试球员" for x in ps), "identity mapping not reflected in player results")
        checks.append("technical ID -> real player confirmation")

        # Human-confirmed eight-dimension assessment. No fake scores are auto-filled.
        assessment_scores = {
            "speed": 78, "endurance": 74, "running": 82, "passing": 71,
            "control": 69, "shooting": 66, "defense": 73, "physical": 76,
        }
        r = client.put(f"/api/projects/{pid}/reviews/assessments/{gid}", json={"scores": assessment_scores, "note": "verification"})
        require(r.status_code == 200 and r.json()["confirmed"] >= 1, "player assessment save failed")
        ps = client.get(f"/api/projects/{pid}/players").json()
        assessed = next((x for x in ps if gid in (x.get("global_ids") or [])), None)
        require(assessed and assessed.get("assessment", {}).get("status") == "confirmed", "player assessment not reflected in card data")
        checks.append("human eight-dimension player assessment")

        # Explicit team semantics.
        r = client.put(f"/api/projects/{pid}/meta", json={"team_labels": {"team_0": "蓝队", "team_1": "橙队", "team_2": "裁判/其他"}})
        require(r.status_code == 200, "team mapping save failed")
        ov = client.get(f"/api/projects/{pid}/overview").json()
        labels = {x.get("label") for x in ov.get("teams", [])}
        require("蓝队" in labels or "橙队" in labels, "team labels not applied")
        checks.append("team semantic mapping")

        require(client.get(f"/api/projects/{pid}/report").status_code == 200, "formal report failed")
        z = client.get(f"/api/projects/{pid}/export.zip")
        require(z.status_code == 200 and len(z.content) > 500, "formal archive failed")
        checks.append("formal report + archive")

        pre = client.get(f"/api/projects/{pid}/preflight")
        require(pre.status_code == 200, "preflight failed")
        checks.append("formal preflight")
    finally:
        client.delete(f"/api/projects/{pid}")

    print("Football Insight V2.3.3 product verification: PASS")
    for item in checks:
        print("  PASS", item)
    print("NOTE: this verifier does not claim a fresh YOLO inference run; that requires models/yolov8x.pt.")


if __name__ == "__main__":
    main()
