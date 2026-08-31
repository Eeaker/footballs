from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import DEFAULT_SETTINGS, PIPELINE_STEPS, PROJECTS_ROOT, DEMO_ROOT, SYSTEM_VERSION


_SAVE_LOCK = threading.RLock()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", text.strip()).strip("-")
    return value[:48] or "match"


def project_dir(project_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", project_id):
        raise ValueError("invalid project id")
    return PROJECTS_ROOT / project_id


def project_json(project_id: str) -> Path:
    return project_dir(project_id) / "project.json"


def _steps() -> list[dict[str, Any]]:
    return [{**step, "state": "pending", "progress": 0, "message": ""} for step in PIPELINE_STEPS]


def _default_pipeline() -> dict[str, Any]:
    return {
        "state": "idle",
        "progress": 0,
        "current_step": None,
        "message": "等待开始",
        "started_at": None,
        "finished_at": None,
        "steps": _steps(),
        "error": None,
        "attempt": 0,
        "resume_from": None,
    }


def normalize_project(project: dict[str, Any]) -> dict[str, Any]:
    """Forward-compatible schema migration for projects created by older app builds."""
    project = deepcopy(project)
    project.setdefault("schema_version", 2)
    project["system_version"] = SYSTEM_VERSION
    project.setdefault("kind", "analysis")
    project.setdefault("status", "draft")
    project.setdefault("video", None)
    project.setdefault("settings", {})
    for key, value in DEFAULT_SETTINGS.items():
        project["settings"].setdefault(key, value)
    project["settings"]["vid_stride"] = 1
    project.setdefault("match", {})
    project["match"].setdefault("competition", "")
    project["match"].setdefault("match_date", "")
    project["match"].setdefault("venue", "")
    project["match"].setdefault("age_group", "")
    project["match"].setdefault("home_team", "主队")
    project["match"].setdefault("away_team", "客队")
    project["match"].setdefault("notes", "")
    project.setdefault("roster", {"status": "missing", "path": None, "count": 0})
    cal = project.setdefault("calibration", {})
    cal.setdefault("mode", "dynamic_rotation_multi_anchor")
    if cal.get("mode") == "dynamic_rotation":
        cal["mode"] = "dynamic_rotation_multi_anchor"
    cal.setdefault("status", "missing")
    cal.setdefault("source", None)
    cal.setdefault("path", None)
    cal.setdefault("reference_path", None)
    cal.setdefault("anchors", [])
    cal.setdefault("validation", None)
    cal.setdefault("message", "请上传已有逐帧动态标定，或创建 1–4 个关键视角锚点。")
    pipe = project.setdefault("pipeline", _default_pipeline())
    defaults = _default_pipeline()
    for key, value in defaults.items():
        pipe.setdefault(key, deepcopy(value))
    old_by_key = {s.get("key"): s for s in pipe.get("steps", [])}
    pipe["steps"] = [
        {**step, **{k: v for k, v in old_by_key.get(step["key"], {}).items() if k not in {"label", "hint"}}}
        for step in _steps()
    ]
    project.setdefault("outputs", {})
    project.setdefault("artifact_manifest", None)
    project.setdefault("run_history", [])
    project.setdefault("review", {
        "pass_review": {"status": "pending", "labeled": 0, "total": 0, "agreement_rate": None},
        "identity_review": {"status": "pending", "flagged_ids": 0},
        "notes": "",
    })
    project.setdefault("created_at", now_iso())
    project.setdefault("updated_at", now_iso())
    return project


def save_project(project: dict[str, Any]) -> dict[str, Any]:
    with _SAVE_LOCK:
        project = normalize_project(project)
        project["updated_at"] = now_iso()
        root = project_dir(project["id"])
        root.mkdir(parents=True, exist_ok=True)
        destination = root / "project.json"
        tmp = root / f"project.json.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        tmp.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            for attempt in range(8):
                try:
                    os.replace(tmp, destination)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        finally:
            tmp.unlink(missing_ok=True)
        return project


def load_project(project_id: str) -> dict[str, Any]:
    path = project_json(project_id)
    if not path.is_file():
        raise FileNotFoundError(project_id)
    project = normalize_project(json.loads(path.read_text(encoding="utf-8")))
    # Persist migrations lazily without changing user data semantics.
    if project.get("system_version") != json.loads(path.read_text(encoding="utf-8")).get("system_version"):
        save_project(project)
    return project


def list_projects() -> list[dict[str, Any]]:
    items = []
    for path in PROJECTS_ROOT.glob("*/project.json"):
        try:
            items.append(normalize_project(json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            continue
    items.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    items.sort(key=lambda item: item.get("kind") == "demo")
    return items


def create_project(name: str, match: dict[str, Any] | None = None) -> dict[str, Any]:
    slug = safe_slug(name)
    project_id = f"{slug}-{uuid.uuid4().hex[:8]}"
    root = project_dir(project_id)
    for sub in ("input", "calibration/anchors", "config", "outputs", "logs", "exports", "reviews"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    project = normalize_project({
        "id": project_id,
        "name": name.strip() or "新比赛",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "kind": "analysis",
        "status": "draft",
        "video": None,
        "settings": dict(DEFAULT_SETTINGS),
        "match": match or {},
        "calibration": {
            "mode": "dynamic_rotation_multi_anchor",
            "status": "missing",
            "source": None,
            "path": None,
            "reference_path": None,
            "anchors": [],
            "validation": None,
            "message": "上传已有动态标定，或在不同旋转视角创建关键帧锚点。",
        },
        "pipeline": _default_pipeline(),
        "outputs": {},
        "artifact_manifest": None,
        "run_history": [],
    })
    return save_project(project)


def delete_project(project_id: str) -> None:
    root = project_dir(project_id)
    if root.exists():
        shutil.rmtree(root)


def mark_stale_running_projects_interrupted() -> None:
    """Threads do not survive a server restart; make that state explicit and retryable."""
    for project in list_projects():
        if project.get("kind") == "demo":
            continue
        if project.get("pipeline", {}).get("state") == "running":
            cur = project["pipeline"].get("current_step")
            resume_step = cur or "tracking"
            for step in project["pipeline"]["steps"]:
                if step["key"] == cur and step.get("state") == "running":
                    step["state"] = "failed"
                    step["message"] = "服务重启中断"
                    resume_step = step["key"]
                    break
            else:
                for step in project["pipeline"]["steps"]:
                    if step.get("state") == "running":
                        step["state"] = "failed"
                        step["message"] = "服务重启中断"
                        resume_step = step["key"]
                        break
            project["pipeline"]["state"] = "interrupted"
            project["pipeline"]["message"] = f"服务重启导致任务中断，可从 {resume_step} 步骤继续"
            project["pipeline"]["current_step"] = resume_step
            project["status"] = "interrupted"
            save_project(project)


def ensure_demo_project() -> dict[str, Any]:
    project_id = "demo-reference-match"
    root = project_dir(project_id)
    for sub in ("input", "calibration/anchors", "config", "outputs", "logs", "exports", "reviews"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    demo = DEMO_ROOT
    demo_video = demo / "source_preview.mp4"
    calibration = demo / "calibration" / "dynamic_calibration.json"
    project = normalize_project({
        "id": project_id,
        "name": "7月24日 · 系统示例比赛",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "kind": "demo",
        "status": "complete",
        "video": {
            "filename": demo_video.name,
            "path": str(demo_video),
            "fps": 24.002,
            "width": 1280,
            "height": 720,
            "frame_count": 568,
            "duration_seconds": 23.665,
            "size_bytes": demo_video.stat().st_size if demo_video.exists() else 0,
            "note": "内置示例片段；指标来自完整比赛分析结果。",
        },
        "settings": {**DEFAULT_SETTINGS, "field_length_m": 45.0, "field_width_m": 25.0},
        "match": {"competition": "示例比赛", "home_team": "蓝队", "away_team": "橙队", "venue": "", "age_group": "U12", "match_date": "", "notes": ""},
        "calibration": {
            "mode": "dynamic_rotation_multi_anchor",
            "status": "ready",
            "source": "verified_example",
            "path": str(calibration),
            "anchors": [],
            "validation": {"passed": True, "accepted_frames": 62204, "total_frames": 62204, "accepted_ratio": 1.0},
            "message": "逐帧动态标定已验证。",
        },
        "pipeline": {
            **_default_pipeline(),
            "state": "complete", "progress": 100, "current_step": "report", "message": "分析完成，可浏览所有结果",
            "finished_at": now_iso(), "steps": [{**step, "state": "complete", "progress": 100, "message": "完成"} for step in PIPELINE_STEPS],
        },
        "outputs": {
            "tracking": str(demo / "tracking"),
            "match_analysis": str(demo / "match_analysis"),
            "number_ocr": str(demo / "number_ocr"),
            "player_cards": str(demo / "player_cards"),
            "report_pdf": str(demo / "nati_report" / "report.pdf"),
            "demo_video": str(demo_video),
        },
    })
    return save_project(project)
