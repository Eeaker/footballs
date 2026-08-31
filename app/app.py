from __future__ import annotations

import json
import mimetypes
import shutil
import zipfile
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ALLOWED_VIDEO_SUFFIXES, STATIC_DIR, SYSTEM_VERSION, MODELS_ROOT, DEFAULT_SETTINGS
from app.services.calibration import build_reference_calibration, dynamic_frame_visualization, normalize_uploaded_dynamic, summarize_dynamic
from app.services.pipeline import cancel_pipeline, start_dynamic_calibration, start_pipeline
from app.services.reporting import build_match_report, build_player_report
from app.services.results import (
    downloadable_files, event_timeline, heatmap_image_path, heatmap_points, highlights, overview, pitch_data, players,
    player_compilation_manifest, player_events, player_visibility_intervals, quality_summary, replay_data, replay_video_path, report_preview_path, summary,
)
from app.services.storage import (
    create_project, delete_project, ensure_demo_project, list_projects, load_project, mark_stale_running_projects_interrupted,
    project_dir, save_project, now_iso,
)
from app.services.system_info import system_status
from app.services.reviews import (
    load_pass_review, save_pass_review_label, load_identity_review, save_identity_mapping,
    identity_merge_candidates, load_player_assessments, save_player_assessment,
    load_player_report_annotation, save_player_report_annotation,
)
from app.services.video import build_player_compilation, build_preview_video, ensure_browser_video, probe_video, read_frame_jpeg
from app.services.video_health import inspect_uploaded_video

app = FastAPI(title="赛场洞察 Football Insight", version=SYSTEM_VERSION)
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
mark_stale_running_projects_interrupted()
ensure_demo_project()


def _build_project_preview(project_id: str, source_path: str) -> None:
    """Create a browser proxy after upload and publish it atomically."""
    output = project_dir(project_id) / "input" / "source_preview.mp4"
    try:
        build_preview_video(source_path, output)
        p = load_project(project_id)
        if str((p.get("video") or {}).get("path")) != str(source_path):
            output.unlink(missing_ok=True)
            return
        static_preview = STATIC_DIR / "project_previews" / f"{project_id}.mp4"
        static_preview.parent.mkdir(parents=True, exist_ok=True)
        static_preview.unlink(missing_ok=True)
        try:
            static_preview.hardlink_to(output)
            p["video"]["preview_url"] = f"/static/project_previews/{project_id}.mp4"
        except OSError:
            # The API endpoint remains available when hard links are unsupported.
            p["video"].pop("preview_url", None)
        p["video"]["preview_path"] = str(output)
        p["video"]["preview_status"] = "ready"
        p["video"].pop("preview_error", None)
        save_project(p)
    except Exception as exc:
        try:
            p = load_project(project_id)
            if str((p.get("video") or {}).get("path")) == str(source_path):
                p["video"]["preview_status"] = "failed"
                p["video"]["preview_error"] = str(exc)
                save_project(p)
        except Exception:
            pass


class SettingsPayload(BaseModel):
    settings: dict[str, Any]


class ProjectMetaPayload(BaseModel):
    name: str | None = None
    competition: str | None = None
    match_date: str | None = None
    venue: str | None = None
    age_group: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    notes: str | None = None
    team_labels: dict[str, str] | None = None


class ReferenceCalibrationPayload(BaseModel):
    frame_index: int
    image_points: list[list[float]]
    world_points: list[list[float]]
    validation_segments: list[dict[str, Any]]
    field_length_m: float
    field_width_m: float
    tolerance_m: float = 0.5


class RunPayload(BaseModel):
    from_step: str = Field(default="tracking", pattern="^(tracking|jersey|events|report)$")


class PassReviewPayload(BaseModel):
    human_is_pass: bool | None = None
    outcome: str = ""
    note: str = ""


class IdentityMappingPayload(BaseModel):
    name: str = ""
    jersey_number: str = ""
    team_id: str = ""
    roster_index: int | None = None
    note: str = ""
    linked_global_ids: list[int] = Field(default_factory=list)


class PlayerAssessmentPayload(BaseModel):
    scores: dict[str, float | None] = Field(default_factory=dict)
    note: str = ""


class PlayerReportAnnotationPayload(BaseModel):
    fields: dict[str, Any] = Field(default_factory=dict)


def _project_preflight(project: dict[str, Any]) -> dict[str, Any]:
    if project.get("kind") == "demo":
        checks = [
            {"key": "video", "label": "比赛视频", "ok": True, "message": "示例素材已就绪"},
            {"key": "calibration", "label": "动态标定", "ok": True, "message": "逐帧映射已验证"},
            {"key": "model", "label": "分析模型", "ok": True, "message": "示例结果已生成"},
            {"key": "output", "label": "结果空间", "ok": True, "message": "结果可直接查看"},
        ]
        return {"ready": True, "checks": checks}
    video = project.get("video") or {}; video_path = Path(str(video.get("path") or ""))
    calibration = project.get("calibration") or {}; calibration_path = Path(str(calibration.get("path") or ""))
    weights_path = Path(str((project.get("settings") or {}).get("weights_path") or ""))
    output_path = project_dir(project["id"]) / "outputs"
    cal_validation = calibration.get("validation") or {}; coverage = cal_validation.get("accepted_ratio")
    cal_message = "逐帧动态标定已就绪"
    if coverage is not None: cal_message += f"（有效覆盖 {float(coverage):.1%}）"
    status = system_status()
    free_gb = status["disk"]["free_bytes"] / (1024**3)
    requested_device = str((project.get("settings") or {}).get("device") or "0").strip().lower()
    compute_ok = requested_device == "cpu" or bool(status.get("gpu", {}).get("available"))
    compute_message = "CPU 模式已选择" if requested_device == "cpu" else (f"GPU 已就绪：{status.get('gpu', {}).get('name') or 'CUDA'}" if compute_ok else "当前选择 GPU，但系统没有检测到可用 CUDA；请安装 GPU 版 PyTorch 或改为 CPU")
    checks = [
        {"key": "video", "label": "比赛视频", "ok": bool(video) and video_path.is_file(), "message": "视频已读取" if bool(video) and video_path.is_file() else "请上传比赛视频"},
        {"key": "calibration", "label": "动态标定", "ok": calibration.get("status") == "ready" and calibration_path.is_file(), "message": cal_message if calibration.get("status") == "ready" and calibration_path.is_file() else "请上传或生成通过验证的动态标定"},
        {"key": "environment", "label": "分析环境", "ok": bool(status.get("readiness", {}).get("inference_dependencies")), "message": "分析依赖已就绪" if status.get("readiness", {}).get("inference_dependencies") else "分析依赖未完整安装，请在系统状态页检查环境"},
        {"key": "compute", "label": "计算设备", "ok": compute_ok, "message": compute_message},
        {"key": "model", "label": "分析模型", "ok": weights_path.is_file(), "message": "分析模型已就绪" if weights_path.is_file() else "请在系统状态页上传 yolov8x.pt，或在高级参数设置模型路径"},
        {"key": "video_tools", "label": "视频工具", "ok": bool(status.get("dependencies", {}).get("cv2")), "message": ("视频读写已就绪" + ("；FFmpeg 可用" if status.get("ffmpeg", {}).get("ready") else "；FFmpeg 未安装但核心分析可运行")) if status.get("dependencies", {}).get("cv2") else "OpenCV 未安装"},
        {"key": "engine", "label": "分析链路", "ok": all((status.get("engine") or {}).values()), "message": "追踪、标定、号码、身份审计、事件和球员卡模块完整" if all((status.get("engine") or {}).values()) else "分析引擎文件不完整，请重新解压正式系统包"},
        {"key": "output", "label": "结果空间", "ok": output_path.is_dir() and free_gb >= 2.0, "message": f"可用空间 {free_gb:.1f} GB" if free_gb >= 2.0 else f"可用空间仅 {free_gb:.1f} GB，建议清理后再运行"},
    ]
    return {"ready": all(item["ok"] for item in checks), "checks": checks}


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/docs/quick", response_class=HTMLResponse)
def quick_doc():
    return (STATIC_DIR / "quick_doc.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health(): return {"ok": True, "version": SYSTEM_VERSION, "time": now_iso()}


@app.get("/api/system/status")
def api_system_status(): return system_status()


@app.get("/api/system/chain")
def api_system_chain():
    path = Path(__file__).resolve().parent.parent / "CHAIN_AUDIT.json"
    if not path.is_file():
        raise HTTPException(404, "链路审计文件不存在")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/system/model")
def upload_model(model: UploadFile = File(...)):
    """Install/replace the default YOLO model from the product UI.

    This is intentionally local-admin functionality for the single-machine
    deployment. The file is staged then atomically moved into models/.
    """
    suffix = Path(model.filename or "").suffix.lower()
    if suffix != ".pt":
        raise HTTPException(400, "模型文件必须是 .pt")
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    target = MODELS_ROOT / "yolov8x.pt"
    staging = MODELS_ROOT / ".yolov8x.pt.uploading"
    try:
        with staging.open("wb") as f:
            shutil.copyfileobj(model.file, f, length=8 * 1024 * 1024)
        if staging.stat().st_size < 1024 * 1024:
            raise ValueError("模型文件过小，请确认上传的是有效权重")
        staging.replace(target)
    except Exception as exc:
        staging.unlink(missing_ok=True)
        raise HTTPException(400, f"模型安装失败：{exc}")
    return system_status()["model"]


@app.get("/api/projects")
def api_projects():
    fields = ("id", "name", "kind", "status", "created_at", "updated_at", "video", "pipeline", "calibration", "match")
    return [{k: p.get(k) for k in fields} for p in list_projects()]


@app.post("/api/projects")
def api_create_project(name: str = Form("新比赛"), home_team: str = Form("主队"), away_team: str = Form("客队"), age_group: str = Form("")):
    return create_project(name, {"home_team": home_team, "away_team": away_team, "age_group": age_group})


@app.get("/api/projects/{project_id}")
def api_project(project_id: str):
    try: return load_project(project_id)
    except FileNotFoundError: raise HTTPException(404, "项目不存在")


@app.put("/api/projects/{project_id}/meta")
def update_project_meta(project_id: str, payload: ProjectMetaPayload):
    p = load_project(project_id)
    if p.get("kind") == "demo": raise HTTPException(400, "示例项目不可修改")
    data = payload.model_dump(exclude_none=True)
    if "name" in data:
        p["name"] = data.pop("name").strip() or p["name"]
    p.setdefault("match", {}).update(data)
    return save_project(p)


@app.delete("/api/projects/{project_id}")
def api_delete_project(project_id: str):
    p = load_project(project_id)
    if p.get("kind") == "demo": raise HTTPException(400, "示例项目不能删除")
    if p.get("pipeline", {}).get("state") == "running": raise HTTPException(400, "请先取消正在运行的任务")
    (STATIC_DIR / "project_previews" / f"{project_id}.mp4").unlink(missing_ok=True)
    delete_project(project_id); return {"ok": True}


@app.get("/api/projects/{project_id}/preflight")
def api_preflight(project_id: str): return _project_preflight(load_project(project_id))


@app.post("/api/projects/{project_id}/video")
def upload_video(project_id: str, background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    p = load_project(project_id)
    if p.get("kind") == "demo": raise HTTPException(400, "示例项目不可修改")
    if p.get("pipeline", {}).get("state") == "running": raise HTTPException(400, "任务运行中不能更换视频")
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES: raise HTTPException(400, "支持 MP4/MOV/AVI/MKV/M4V")
    input_dir = project_dir(project_id) / "input"
    for old in input_dir.glob("source.*"): old.unlink(missing_ok=True)
    (input_dir / "source_preview.mp4").unlink(missing_ok=True)
    (STATIC_DIR / "project_previews" / f"{project_id}.mp4").unlink(missing_ok=True)
    dest = input_dir / ("source" + suffix)
    with dest.open("wb") as f: shutil.copyfileobj(video.file, f, length=8*1024*1024)
    try: meta = probe_video(dest)
    except Exception as e:
        dest.unlink(missing_ok=True); raise HTTPException(400, str(e))
    meta["original_filename"] = video.filename
    try:
        meta["health"] = inspect_uploaded_video(dest, sample_pairs=32)
    except Exception as health_exc:
        meta["health"] = {
            "status": "warning", "motion_type": "unknown", "calibration_mode": "dynamic_keyframes",
            "product_calibration_policy": "dynamic_multi_anchor",
            "product_note": "视频体检未完整完成，但正式系统仍要求多锚点逐帧动态标定。",
            "warning": str(health_exc),
        }
    meta["preview_status"] = "building"
    p["video"] = meta; p["status"] = "configured"
    p["calibration"].update(status="missing", source=None, path=None, reference_path=None, anchors=[], validation=None,
        message="视频已更新，请重新上传或创建动态标定。")
    p["outputs"] = {}; p["artifact_manifest"] = None
    save_project(p)
    background_tasks.add_task(_build_project_preview, project_id, str(dest))
    return p


@app.post("/api/projects/{project_id}/roster")
def upload_roster(project_id: str, roster: UploadFile = File(...)):
    p=load_project(project_id)
    if p.get("kind") == "demo": raise HTTPException(400,"示例项目不可修改")
    suffix=Path(roster.filename or "").suffix.lower()
    if suffix not in {".csv", ".json"}: raise HTTPException(400,"名单支持 CSV 或 JSON")
    dest=project_dir(project_id)/"input"/("roster"+suffix)
    with dest.open("wb") as f: shutil.copyfileobj(roster.file,f)
    count=0
    try:
        if suffix==".json":
            data=json.loads(dest.read_text(encoding="utf-8-sig")); count=len(data if isinstance(data,list) else data.get("players",[]))
        else:
            import csv
            with dest.open(encoding="utf-8-sig",newline="") as f: count=sum(1 for _ in csv.DictReader(f))
    except Exception as e:
        dest.unlink(missing_ok=True); raise HTTPException(400,f"名单读取失败：{e}")
    p["roster"]={"status":"ready","path":str(dest),"count":count,"filename":roster.filename}; save_project(p); return p["roster"]


@app.put("/api/projects/{project_id}/settings")
def update_settings(project_id: str, payload: SettingsPayload):
    p = load_project(project_id)
    if p.get("kind") == "demo": raise HTTPException(400, "示例项目参数已锁定")
    if p.get("pipeline", {}).get("state") == "running": raise HTTPException(400, "任务运行中不能修改参数")
    allowed = set(p["settings"])
    for key, val in payload.settings.items():
        if key in allowed: p["settings"][key] = val
    p["settings"]["vid_stride"] = 1
    save_project(p); return p["settings"]


@app.get("/api/projects/{project_id}/settings/export")
def export_settings(project_id: str):
    p = load_project(project_id)
    portable = {k: v for k, v in (p.get("settings") or {}).items() if k != "weights_path"}
    payload = {
        "schema_version": 1, "system": "Football Insight", "system_version": SYSTEM_VERSION,
        "project_name": p.get("name"), "settings": portable,
        "note": "weights_path is intentionally omitted so this template remains portable across Windows machines.",
    }
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return Response(data, media_type="application/json", headers={"Content-Disposition": 'attachment; filename="football_insight_parameters.json"'})


@app.post("/api/projects/{project_id}/settings/import")
def import_settings(project_id: str, config: UploadFile = File(...)):
    p = load_project(project_id)
    if p.get("kind") == "demo": raise HTTPException(400, "示例项目参数已锁定")
    if p.get("pipeline", {}).get("state") == "running": raise HTTPException(400, "任务运行中不能修改参数")
    try:
        data = json.loads(config.file.read().decode("utf-8-sig"))
    except Exception:
        raise HTTPException(400, "参数模板必须是 JSON")
    incoming = data.get("settings") if isinstance(data, dict) and isinstance(data.get("settings"), dict) else data
    if not isinstance(incoming, dict): raise HTTPException(400, "参数模板缺少 settings 对象")
    allowed = set(DEFAULT_SETTINGS)
    changed = []
    for key, val in incoming.items():
        if key in allowed and key not in {"weights_path", "vid_stride"}:
            p["settings"][key] = val; changed.append(key)
    p["settings"]["vid_stride"] = 1
    save_project(p)
    return {"settings": p["settings"], "imported_keys": sorted(changed), "ignored_keys": sorted(set(incoming) - set(changed))}


@app.get("/api/projects/{project_id}/source-video")
def source_video(project_id: str):
    p = load_project(project_id); path = Path(str((p.get("video") or {}).get("path") or ""))
    if not path.is_file(): raise HTTPException(404, "比赛视频不存在")
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "video/mp4")


@app.get("/api/projects/{project_id}/preview-video")
def preview_video(project_id: str):
    p = load_project(project_id); video = p.get("video") or {}
    preview = Path(str(video.get("preview_path") or ""))
    if video.get("preview_status") == "ready" and preview.is_file():
        return FileResponse(preview, media_type="video/mp4")
    # Older projects may already have a proxy created by a newer app build but
    # lack the metadata fields because their project.json predates them.
    for candidate in (project_dir(project_id) / "input" / "source_preview.mp4", STATIC_DIR / "project_previews" / f"{project_id}.mp4"):
        if candidate.is_file():
            return FileResponse(candidate, media_type="video/mp4")
    source = Path(str(video.get("path") or ""))
    if not source.is_file(): raise HTTPException(404, "比赛视频不存在")
    return FileResponse(source, media_type=mimetypes.guess_type(source.name)[0] or "video/mp4")


@app.get("/api/projects/{project_id}/frame")
def frame(project_id: str, frame_index: int = 0):
    p = load_project(project_id)
    if not p.get("video"): raise HTTPException(400, "请先上传视频")
    try: data, _, _ = read_frame_jpeg(p["video"]["path"], frame_index)
    except Exception as e: raise HTTPException(400, str(e))
    # A project video is immutable between uploads. The UI adds a video revision
    # query token, so decoded timeline frames can be reused while scrubbing.
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@app.post("/api/projects/{project_id}/calibration/upload")
def upload_calibration(project_id: str, config: UploadFile = File(...)):
    p = load_project(project_id)
    if p.get("kind") == "demo": raise HTTPException(400, "示例项目不可修改")
    if not p.get("video"): raise HTTPException(400, "请先上传视频")
    try: data = json.loads(config.file.read().decode("utf-8-sig"))
    except Exception: raise HTTPException(400, "标定文件必须是 JSON")
    out = project_dir(project_id) / "calibration" / "dynamic_calibration.json"
    try: validation = normalize_uploaded_dynamic(data, p["video"], out, float(p["settings"].get("dynamic_min_coverage",0.8)))
    except Exception as e: raise HTTPException(400, str(e))
    p["calibration"].update(
        status="ready", source="uploaded_dynamic", path=str(out), reference_path=None, anchors=[],
        source_filename=Path(config.filename or "dynamic_calibration.json").name,
        uploaded_at=now_iso(), validation=validation,
        message="已加载与当前视频匹配的逐帧动态标定",
    )
    save_project(p); return p["calibration"]


@app.get("/api/projects/{project_id}/calibration/download")
def download_calibration(project_id: str):
    p = load_project(project_id)
    path = Path(str((p.get("calibration") or {}).get("path") or ""))
    if (p.get("calibration") or {}).get("status") != "ready" or not path.is_file():
        raise HTTPException(404, "动态标定尚未生成")
    return FileResponse(path, media_type="application/json", filename=f"{project_id}_dynamic_calibration.json")


@app.get("/api/projects/{project_id}/calibration/visualization")
def calibration_visualization(project_id: str):
    """Return the small, safe subset needed to render an uploaded calibration."""
    p = load_project(project_id)
    calibration = p.get("calibration") or {}
    path = Path(str(calibration.get("path") or ""))
    if calibration.get("status") != "ready" or not path.is_file():
        raise HTTPException(404, "动态标定尚未生成")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(400, f"动态标定无法读取：{exc}")

    registration = data.get("dynamic_registration") or {}
    frame_index = data.get("calibration_proc_idx")
    if frame_index is None:
        frame_index = registration.get("reference_proc_idx")
    anchors = data.get("anchors") or []
    if frame_index is None and anchors:
        frame_index = anchors[0].get("proc_idx", anchors[0].get("frame_index"))
    frame_index = max(0, min(int(frame_index or 0), int(p["video"]["frame_count"]) - 1))

    image_points = data.get("image_points") or []
    world_points = data.get("world_points_m") or data.get("world_points") or []
    validation_segments = data.get("validation_segments") or []
    if (not image_points or not world_points) and anchors:
        anchor = min(
            anchors,
            key=lambda row: abs(int(row.get("proc_idx", row.get("frame_index", frame_index))) - frame_index),
        )
        image_points = anchor.get("image_points") or []
        world_points = anchor.get("world_points_m") or anchor.get("world_points") or []
        validation_segments = anchor.get("validation_segments") or validation_segments

    safe_segments = []
    validation_results = (data.get("validation") or {}).get("results") or []
    for index, row in enumerate(validation_segments):
        points = row.get("image_points") or [row.get("p1"), row.get("p2")]
        if len(points) != 2 or points[0] is None or points[1] is None:
            continue
        result = validation_results[index] if index < len(validation_results) else {}
        safe_segments.append({
            "name": row.get("name") or f"validation_{index + 1}",
            "p1": points[0], "p2": points[1],
            "known_length_m": row.get("known_length_m", row.get("length_m")),
            "measured_length_m": result.get("measured_length_m"),
            "absolute_error_m": result.get("absolute_error_m"),
            "passed": result.get("passed"),
        })
    return {
        "frame_index": frame_index,
        "image_points": image_points,
        "world_points_m": world_points,
        "validation_segments": safe_segments,
        "field_bounds_m": data.get("field_bounds_m") or {},
        "summary": summarize_dynamic(path),
    }


@app.get("/api/projects/{project_id}/calibration/frame-visualization")
def calibration_frame_visualization(project_id: str, frame_index: int = 0):
    p = load_project(project_id)
    calibration = p.get("calibration") or {}
    path = Path(str(calibration.get("path") or ""))
    if calibration.get("status") != "ready" or not path.is_file():
        raise HTTPException(404, "逐帧动态标定尚未就绪")
    try:
        return dynamic_frame_visualization(path, frame_index)
    except Exception as exc:
        raise HTTPException(400, f"无法读取该帧的动态标定：{exc}") from exc


@app.post("/api/projects/{project_id}/calibration/anchors")
def add_calibration_anchor(project_id: str, payload: ReferenceCalibrationPayload):
    p = load_project(project_id)
    if not p.get("video"): raise HTTPException(400, "请先上传视频")
    if p.get("kind") == "demo": raise HTTPException(400,"示例项目不可修改")
    anchors=p["calibration"].get("anchors") or []
    if len(anchors)>=6: raise HTTPException(400,"最多支持 6 个标定锚点")
    anchor_id=f"anchor_{payload.frame_index:07d}"
    out=project_dir(project_id)/"calibration"/"anchors"/f"{anchor_id}.json"
    try:
        data=build_reference_calibration(video=p["video"], frame_index=payload.frame_index, image_points=payload.image_points, world_points=payload.world_points,
            validation_segments=payload.validation_segments, field_length_m=payload.field_length_m, field_width_m=payload.field_width_m, tolerance_m=payload.tolerance_m, output=out)
    except Exception as e: raise HTTPException(400,str(e))
    passed=bool(data["validation"]["passed"])
    # Replace an anchor on the same frame rather than duplicate it.
    anchors=[a for a in anchors if int(a.get("frame_index",-1))!=payload.frame_index]
    anchors.append({"id":anchor_id,"frame_index":payload.frame_index,"path":str(out),"passed":passed,"validation":data["validation"]})
    anchors.sort(key=lambda a:int(a["frame_index"]))
    p["settings"]["field_length_m"]=payload.field_length_m; p["settings"]["field_width_m"]=payload.field_width_m
    p["calibration"].update(status="anchors_ready" if any(a["passed"] for a in anchors) else "anchor_failed", source="manual_multi_anchor", path=None, anchors=anchors,
        validation=None, message=(f"已保存 {len(anchors)} 个视角锚点，可继续增加其他旋转视角或生成全片动态标定" if passed else "该锚点独立验证未通过，请重新点选"))
    save_project(p); return p["calibration"]


# Backward-compatible alias used by V1 front-end bookmarks.
@app.post("/api/projects/{project_id}/calibration/reference")
def reference_calibration_alias(project_id: str, payload: ReferenceCalibrationPayload):
    return add_calibration_anchor(project_id, payload)


@app.delete("/api/projects/{project_id}/calibration/anchors/{anchor_id}")
def delete_calibration_anchor(project_id: str, anchor_id: str):
    p=load_project(project_id); anchors=p["calibration"].get("anchors") or []; hit=[a for a in anchors if a.get("id")==anchor_id]
    if not hit: raise HTTPException(404,"标定锚点不存在")
    Path(str(hit[0].get("path") or "")).unlink(missing_ok=True)
    anchors=[a for a in anchors if a.get("id")!=anchor_id]; p["calibration"]["anchors"]=anchors; p["calibration"].update(path=None,validation=None)
    p["calibration"]["status"]="anchors_ready" if any(a.get("passed") for a in anchors) else "missing"
    p["calibration"]["message"]="锚点已删除，请重新生成全片动态标定" if anchors else "请创建动态标定锚点"
    save_project(p); return p["calibration"]


@app.post("/api/projects/{project_id}/calibration/expand")
def expand_calibration(project_id: str):
    p = load_project(project_id)
    if p.get("kind") == "demo": raise HTTPException(400, "示例项目不可修改")
    anchors=[a for a in p["calibration"].get("anchors",[]) if a.get("passed")]
    if not anchors and p["calibration"].get("status") != "reference_ready": raise HTTPException(400, "请先完成至少 1 个通过验证的视角锚点")
    try: start_dynamic_calibration(project_id)
    except Exception as e: raise HTTPException(500, f"动态标定启动失败：{e}")
    return load_project(project_id)["calibration"]


@app.post("/api/projects/{project_id}/run")
def run_project(project_id: str, payload: RunPayload | None = None):
    p = load_project(project_id)
    if p.get("kind") == "demo": return {"ok": True, "message": "示例项目已完成"}
    preflight = _project_preflight(p)
    if not preflight["ready"]:
        missing = [item["message"] for item in preflight["checks"] if not item["ok"]]
        raise HTTPException(400, "开始分析前请完成：" + "；".join(missing))
    from_step=(payload.from_step if payload else "tracking")
    try: start_pipeline(project_id, from_step=from_step)
    except Exception as e: raise HTTPException(400, str(e))
    return {"ok": True, "from_step": from_step}


@app.post("/api/projects/{project_id}/cancel")
def cancel_project(project_id: str): cancel_pipeline(project_id); return {"ok": True}


@app.get("/api/projects/{project_id}/summary")
def api_summary(project_id: str): return summary(load_project(project_id))

@app.get("/api/projects/{project_id}/overview")
def api_overview(project_id: str): return overview(load_project(project_id))

@app.get("/api/projects/{project_id}/players")
def api_players(project_id: str): return players(load_project(project_id))

@app.get("/api/projects/{project_id}/players/{player_index}/heatmap")
def api_heatmap(project_id: str, player_index: int):
    p=load_project(project_id); ps=players(p)
    if player_index<0 or player_index>=len(ps): raise HTTPException(404,"球员不存在")
    ids = ps[player_index].get("global_ids", [])
    image = heatmap_image_path(p, ids)
    return {
        "field": pitch_data(p,5)["field"],
        "points": heatmap_points(p, ids),
        "image_url": f"/api/projects/{project_id}/players/{player_index}/heatmap.png" if image else None,
    }

@app.get("/api/projects/{project_id}/players/{player_index}/heatmap.png")
def api_heatmap_image(project_id: str, player_index: int):
    p=load_project(project_id); ps=players(p)
    if player_index<0 or player_index>=len(ps): raise HTTPException(404,"球员不存在")
    path = heatmap_image_path(p, ps[player_index].get("global_ids", []))
    if path is None: raise HTTPException(404,"该球员暂无可用热力图")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

@app.get("/api/projects/{project_id}/players/{player_index}/mosaic")
def api_identity_mosaic(project_id: str, player_index: int):
    p=load_project(project_id); ps=players(p)
    if player_index<0 or player_index>=len(ps): raise HTTPException(404,"球员不存在")
    gids = ps[player_index].get("global_ids", [])
    if not gids: raise HTTPException(404,"该球员无关联技术 ID")
    root = project_dir(project_id) / "outputs" / "tracking" / "identity_mosaics"
    manifest_path = root / "identity_mosaics_manifest.json"
    if not manifest_path.is_file(): raise HTTPException(404,"身份拼图未生成")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception: raise HTTPException(500,"身份拼图清单读取失败")
    items = manifest.get("items", [])
    gid_set = set(int(g) for g in gids)
    matched = [item for item in items if int(item.get("global_id", -1)) in gid_set]
    if not matched: raise HTTPException(404,"该球员无身份拼图")
    results = []
    for item in matched:
        mosaic_path = root / item.get("mosaic_file", "")
        if mosaic_path.is_file():
            results.append({
                "global_id": item.get("global_id"),
                "mosaic_url": f"/api/projects/{project_id}/players/{player_index}/mosaic/{item.get('global_id')}",
                "sample_count": item.get("sample_count", 0),
                "visible_seconds": item.get("visible_seconds", 0),
            })
    return {"mosaics": results}

@app.get("/api/projects/{project_id}/players/{player_index}/mosaic/{global_id}")
def api_identity_mosaic_image(project_id: str, player_index: int, global_id: int):
    p=load_project(project_id); ps=players(p)
    if player_index<0 or player_index>=len(ps): raise HTTPException(404,"球员不存在")
    root = project_dir(project_id) / "outputs" / "tracking" / "identity_mosaics"
    manifest_path = root / "identity_mosaics_manifest.json"
    if not manifest_path.is_file(): raise HTTPException(404,"身份拼图未生成")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception: raise HTTPException(500,"身份拼图清单读取失败")
    for item in manifest.get("items", []):
        if int(item.get("global_id", -1)) == global_id:
            mosaic_path = root / item.get("mosaic_file", "")
            if mosaic_path.is_file():
                return FileResponse(mosaic_path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
    raise HTTPException(404,"身份拼图不存在")

@app.get("/api/projects/{project_id}/players/{player_index}/events")
def api_player_events(project_id: str, player_index: int):
    p=load_project(project_id); ps=players(p)
    if player_index<0 or player_index>=len(ps): raise HTTPException(404,"球员不存在")
    return player_events(p, ps[player_index].get("global_ids",[]))

@app.get("/api/projects/{project_id}/players/{player_index}/compilation.mp4")
def api_player_compilation(project_id: str, player_index: int):
    p=load_project(project_id); ps=players(p)
    if player_index<0 or player_index>=len(ps): raise HTTPException(404,"球员不存在")
    video=p.get("video") or {}; source=Path(str(video.get("preview_path") or ""))
    if not source.is_file():
        legacy=project_dir(project_id)/"input"/"source_preview.mp4"
        source=legacy if legacy.is_file() else Path(str(video.get("path") or ""))
    gids=[int(value) for value in ps[player_index].get("global_ids",[])]
    intervals=player_visibility_intervals(p,gids)
    identity_suffix="-".join(map(str,gids)) or "unknown"
    target=project_dir(project_id)/"outputs"/"player_compilations"/f"player_{player_index+1:03d}_ids_{identity_suffix}.mp4"
    try: path=build_player_compilation(source,intervals,target)
    except RuntimeError as exc: raise HTTPException(500,str(exc))
    return FileResponse(path,media_type="video/mp4",filename=path.name,content_disposition_type="inline")


@app.get("/api/projects/{project_id}/players/{player_index}/compilation-manifest")
def api_player_compilation_manifest(project_id: str, player_index: int):
    p=load_project(project_id); ps=players(p)
    if player_index<0 or player_index>=len(ps): raise HTTPException(404,"球员不存在")
    return player_compilation_manifest(p,ps[player_index].get("global_ids",[]))

@app.get("/api/projects/{project_id}/players/{player_index}/report", response_class=HTMLResponse)
def api_player_report(project_id: str, player_index: int):
    p=load_project(project_id)
    try: path,_=build_player_report(p,project_dir(project_id)/"outputs",player_index)
    except IndexError: raise HTTPException(404,"球员不存在")
    except Exception as exc: raise HTTPException(500,f"球员报告生成失败：{exc}")
    return path.read_text(encoding="utf-8")

@app.get("/api/projects/{project_id}/players/{player_index}/report.pdf")
def api_player_report_pdf(project_id: str, player_index: int):
    p=load_project(project_id)
    try: _,path=build_player_report(p,project_dir(project_id)/"outputs",player_index,make_pdf=True)
    except IndexError: raise HTTPException(404,"球员不存在")
    except Exception as exc: raise HTTPException(500,f"球员 PDF 生成失败：{exc}")
    if path is None: raise HTTPException(500,"球员 PDF 未生成")
    return FileResponse(path,media_type="application/pdf",filename=path.name)

@app.get("/api/projects/{project_id}/pitch")
def api_pitch(project_id: str): return pitch_data(load_project(project_id))

@app.get("/api/projects/{project_id}/replay")
def api_replay(project_id: str, max_frames: int | None = None, start_frame: int | None = None, frame_count: int | None = None):
    try:
        return replay_data(load_project(project_id), max_frames=max_frames, start_frame=start_frame, frame_count=frame_count)
    except Exception as exc:
        return {"field": {"length_m": 45.0, "width_m": 25.0}, "fps": 30.0, "duration_seconds": 0, "frame_step": 1, "frames": [], "passes": [], "ball_observations": [], "total_frames": 0, "window_start": 0, "window_end": 0, "sampling_mode": "source_frame", "error": str(exc)}

@app.get("/api/projects/{project_id}/replay.mp4")
def api_replay_video(project_id: str):
    path=replay_video_path(load_project(project_id))
    if path is None: raise HTTPException(404,"2D 回放视频尚未生成")
    cache=project_dir(project_id)/"browser_media"/"replay.mp4"
    try: browser_path=ensure_browser_video(path,cache)
    except RuntimeError as exc: raise HTTPException(500,str(exc))
    return FileResponse(browser_path,media_type="video/mp4",filename=browser_path.name,content_disposition_type="inline")

@app.get("/api/projects/{project_id}/events")
def api_events(project_id: str, limit: int = 500): return event_timeline(load_project(project_id),limit)

@app.get("/api/projects/{project_id}/quality")
def api_quality(project_id: str): return quality_summary(load_project(project_id))


@app.get("/api/projects/{project_id}/reviews/passes")
def api_pass_reviews(project_id: str):
    p = load_project(project_id)
    return load_pass_review(p)


@app.put("/api/projects/{project_id}/reviews/passes/{key}")
def api_save_pass_review(project_id: str, key: str, payload: PassReviewPayload):
    p = load_project(project_id)
    try:
        state = save_pass_review_label(p, key, payload.human_is_pass, payload.outcome, payload.note)
    except KeyError:
        raise HTTPException(404, "复核样本不存在")
    except ValueError as e:
        raise HTTPException(400, str(e))
    p.setdefault("review", {})["pass_review"] = {
        "status": state["status"], "labeled": state["labeled"], "total": state["total"],
        "agreement_rate": state["agreement_rate"],
    }
    save_project(p)
    return state


@app.get("/api/projects/{project_id}/reviews/identities")
def api_identity_reviews(project_id: str):
    return load_identity_review(load_project(project_id))


@app.get("/api/projects/{project_id}/reviews/identities/{global_id}/merge-candidates")
def api_identity_merge_candidates(project_id: str, global_id: int):
    try: return identity_merge_candidates(load_project(project_id), global_id)
    except KeyError: raise HTTPException(404,"技术 ID 不存在")


@app.put("/api/projects/{project_id}/reviews/identities/{global_id}")
def api_save_identity_mapping(project_id: str, global_id: int, payload: IdentityMappingPayload):
    p = load_project(project_id)
    try:
        state = save_identity_mapping(
            p, global_id, name=payload.name, jersey_number=payload.jersey_number, team_id=payload.team_id,
            roster_index=payload.roster_index, note=payload.note, linked_global_ids=payload.linked_global_ids,
        )
    except KeyError:
        raise HTTPException(404, "技术 ID 不存在")
    except ValueError as e:
        raise HTTPException(400, str(e))
    p.setdefault("review", {})["identity_review"] = {
        "status": state["status"], "confirmed": state["confirmed"], "total": state["total"],
    }
    save_project(p)
    return state


@app.get("/api/projects/{project_id}/reviews/assessments")
def api_player_assessments(project_id: str):
    return load_player_assessments(load_project(project_id))


@app.put("/api/projects/{project_id}/reviews/assessments/{global_id}")
def api_save_player_assessment(project_id: str, global_id: int, payload: PlayerAssessmentPayload):
    p = load_project(project_id)
    try:
        state = save_player_assessment(p, global_id, scores=payload.scores, note=payload.note)
    except KeyError:
        raise HTTPException(404, "技术 ID 不存在")
    except ValueError as e:
        raise HTTPException(400, str(e))
    p.setdefault("review", {})["player_assessment"] = {
        "status": state["status"], "confirmed": state["confirmed"], "total": state["total"],
    }
    save_project(p)
    return state


@app.get("/api/projects/{project_id}/reviews/player-report/{global_id}")
def api_player_report_annotation(project_id: str, global_id: int):
    try: return load_player_report_annotation(load_project(project_id), global_id)
    except KeyError: raise HTTPException(404,"技术 ID 不存在")


@app.put("/api/projects/{project_id}/reviews/player-report/{global_id}")
def api_save_player_report_annotation(project_id: str, global_id: int, payload: PlayerReportAnnotationPayload):
    try: return save_player_report_annotation(load_project(project_id), global_id, payload.fields)
    except KeyError: raise HTTPException(404,"技术 ID 不存在")
    except ValueError as exc: raise HTTPException(400,str(exc))

@app.get("/api/projects/{project_id}/highlights")
def api_highlights(project_id: str):
    hs=highlights(load_project(project_id)); return [{k:v for k,v in h.items() if k!="path"}|{"index":i} for i,h in enumerate(hs)]

@app.get("/api/projects/{project_id}/highlight/{index}")
def api_highlight_media(project_id: str, index: int):
    p=load_project(project_id); hs=highlights(p)
    if index<0 or index>=len(hs): raise HTTPException(404,"视频不存在")
    source=Path(hs[index]["path"]); cache=project_dir(project_id)/"browser_media"/"highlights"/source.name
    try: path=ensure_browser_video(source,cache)
    except RuntimeError as exc: raise HTTPException(500,str(exc))
    return FileResponse(path, media_type="video/mp4", filename=path.name, content_disposition_type="inline")

@app.get("/api/projects/{project_id}/report", response_class=HTMLResponse)
def api_report(project_id: str):
    p=load_project(project_id)
    if p.get("kind") != "demo" and (p.get("pipeline",{}).get("state")=="complete" or p.get("status")=="complete"):
        try:
            build_match_report(p, project_dir(project_id)/"outputs")
        except Exception:
            pass
    path=report_preview_path(p)
    if path is None: raise HTTPException(404,"比赛报告尚未生成")
    return path.read_text(encoding="utf-8")

@app.get("/api/projects/{project_id}/files")
def api_files(project_id: str):
    return [{"index":i,"label":r["label"],"category":r.get("category"),"filename":r.get("filename"),"size_bytes":r["size_bytes"]} for i,r in enumerate(downloadable_files(load_project(project_id)))]

@app.get("/api/projects/{project_id}/file/{index}")
def api_file(project_id: str,index:int):
    rows=downloadable_files(load_project(project_id))
    if index<0 or index>=len(rows): raise HTTPException(404,"文件不存在")
    path=Path(rows[index]["path"]); return FileResponse(path,media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",filename=path.name)

@app.get("/api/projects/{project_id}/export.zip")
def export_zip(project_id: str):
    p=load_project(project_id); root=project_dir(project_id); export_dir=root/"exports"; export_dir.mkdir(parents=True,exist_ok=True); target=export_dir/f"{project_id}_results.zip"
    if p.get("kind") != "demo" and (p.get("pipeline",{}).get("state")=="complete" or p.get("status")=="complete"):
        try: build_match_report(p, root/"outputs")
        except Exception: pass
    # Rebuild to guarantee the archive reflects the current run. Source video is intentionally excluded.
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        z.writestr("project_summary.json",json.dumps(summary(p),ensure_ascii=False,indent=2))
        z.writestr("project_metadata.json",json.dumps({k:p.get(k) for k in ("id","name","match","settings","calibration","pipeline","outputs")},ensure_ascii=False,indent=2))
        for folder_name in ("calibration","config","outputs","reviews"):
            folder=root/folder_name
            if not folder.exists(): continue
            for path in folder.rglob("*"):
                if path.is_file(): z.write(path,arcname=str(Path(folder_name)/path.relative_to(folder)))
        # Keep roster / metadata inputs, but deliberately exclude the usually huge source video.
        input_dir=root/"input"
        if input_dir.exists():
            for path in input_dir.glob("roster.*"):
                if path.is_file(): z.write(path,arcname=str(Path("input")/path.name))
    return FileResponse(target,media_type="application/zip",filename=target.name)

@app.get("/api/projects/{project_id}/runs")
def api_run_history(project_id: str):
    p = load_project(project_id)
    return {"project_id": project_id, "pipeline": p.get("pipeline", {}), "runs": list(reversed(p.get("run_history", [])))}


@app.get("/api/projects/{project_id}/log")
def api_log(project_id: str):
    path=project_dir(project_id)/"logs"/"pipeline.log"
    if not path.is_file(): return {"text":""}
    text=path.read_text(encoding="utf-8",errors="replace"); return {"text":"\n".join(text.splitlines()[-200:])}
