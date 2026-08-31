from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.storage import load_project, project_dir


def exists_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify that a formal project completed the real pipeline on this machine.")
    ap.add_argument("project_id")
    args = ap.parse_args()

    try:
        p = load_project(args.project_id)
    except FileNotFoundError:
        print("FAIL project not found")
        return 2

    root = project_dir(args.project_id)
    out = root / "outputs"
    checks = []
    def add(name: str, ok: bool, detail: str = ""):
        checks.append((name, bool(ok), detail))

    add("formal project", p.get("kind") == "analysis", str(p.get("kind")))
    video = Path(str((p.get("video") or {}).get("path") or ""))
    add("source video", exists_nonempty(video), str(video))
    cal = Path(str((p.get("calibration") or {}).get("path") or ""))
    add("dynamic calibration", p.get("calibration", {}).get("status") == "ready" and exists_nonempty(cal), str(cal))
    add("pipeline complete", p.get("pipeline", {}).get("state") == "complete", str(p.get("pipeline", {}).get("state")))

    history = p.get("run_history", [])
    real_runs = [r for r in history if r.get("state") == "complete" and r.get("run_id") and "fixture" not in str(r).lower()]
    add("completed run history", bool(real_runs), f"completed_runs={len(real_runs)}")

    required = {
        "tracking MOT": out / "tracking" / "tracking" / "tracking_mot.txt",
        "tracking video": out / "tracking" / "tracking" / "tracking_vis.mp4",
        "jersey OCR": out / "number_ocr" / "jersey_number_results.csv",
        "running summary": out / "match_analysis" / "metric_running" / "player_running_summary.csv",
        "running timeseries": out / "match_analysis" / "metric_running" / "player_running_timeseries.csv",
        "pass events": out / "match_analysis" / "analysis" / "pass_events.csv",
        "possession": out / "match_analysis" / "analysis" / "possession_intervals.csv",
        "events": out / "events_for_annotation.json",
        "2D replay": out / "metric_pitch_replay.mp4",
        "match report": out / "match_report.html",
        "artifact manifest": out / "artifact_manifest.json",
    }
    for name, path in required.items():
        add(name, exists_nonempty(path), str(path))

    highlights = out / "highlights"
    high_files = list(highlights.rglob("*.mp4")) if highlights.is_dir() else []
    add("TARGET/highlight media", any(x.stat().st_size > 0 for x in high_files), f"clips={len(high_files)}")

    cards = out / "player_cards"
    card_files = list(cards.rglob("*")) if cards.is_dir() else []
    add("player cards", any(x.is_file() and x.stat().st_size > 0 for x in card_files), f"files={sum(x.is_file() for x in card_files)}")

    ok = all(x[1] for x in checks)
    print(f"Fresh project verification: {'PASS' if ok else 'FAIL'}")
    for name, passed, detail in checks:
        print(f"  {'PASS' if passed else 'FAIL':<4} {name:<24} {detail}")
    if ok:
        result = {
            "schema_version": 1,
            "project_id": args.project_id,
            "fresh_end_to_end_pipeline": True,
            "note": "This verifies pipeline completion and required artifacts, not algorithmic accuracy. Accuracy still requires human/GT acceptance.",
        }
        target = root / "reviews" / "fresh_pipeline_verification.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("  RECORD", target)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
