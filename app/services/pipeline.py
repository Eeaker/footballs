from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Any

import yaml

from app.config import SYSTEM_ROOT, ENGINE_ROOT, PIPELINE_STEPS
from app.services.calibration import summarize_dynamic
from app.services.storage import load_project, project_dir, save_project, now_iso

_RUNNERS: dict[str, "PipelineRunner"] = {}
_CALIBRATORS: dict[str, threading.Thread] = {}
_LOCK = threading.Lock()
_STAGE_ORDER = [step["key"] for step in PIPELINE_STEPS]
_MAX_CONCURRENT_PIPELINES = max(1, int(os.getenv("FOOTBALL_INSIGHT_MAX_JOBS", "1")))


def _tracking_progress(line: str, total_frames: int) -> tuple[int, str] | None:
    """Translate tracking-engine milestones into stable product progress."""
    total = max(1, int(total_frames))
    match = re.search(r"处理到帧\s*([\d,]+)", line)
    if match:
        frame = int(match.group(1).replace(",", ""))
        progress = min(78, 5 + int(frame / total * 73))
        return progress, f"正在追踪球员与足球… {min(frame, total):,}/{total:,} 帧"
    if "[Field]" in line:
        return 79, "正在检查场内轨迹…"
    if "[Stage 2]" in line:
        return 81, "正在进行全局身份重关联…"
    match = re.search(r"渲染到帧\s*([\d,]+)", line)
    if match:
        frame = int(match.group(1).replace(",", ""))
        progress = min(94, 83 + int(frame / total * 11))
        return progress, f"正在渲染追踪回放… {min(frame, total):,}/{total:,} 帧"
    if "[Stage 3]" in line:
        return 83, "正在渲染追踪回放…"
    if "[Stage 4]" in line:
        return 94, "正在整理追踪事件与片段…"
    return None


def _player_card_progress(line: str) -> tuple[int, str] | None:
    """Map formal player-card delivery counts into the report-stage band."""
    match = re.search(r"FT player cards:\s*players=(\d+)/(\d+)", line)
    if not match:
        return None
    done, total = int(match.group(1)), max(1, int(match.group(2)))
    progress = min(40, 5 + int(min(done, total) / total * 35))
    return progress, f"正在生成球员卡与事件片段… {min(done, total)}/{total} 人"


def _set_step(project: dict, key: str, *, state: str, progress: int, message: str | None = None) -> None:
    for step in project["pipeline"]["steps"]:
        if step["key"] == key:
            step["state"] = state
            step["progress"] = int(progress)
            if message is not None:
                step["message"] = message
    project["pipeline"]["current_step"] = key
    if message:
        project["pipeline"]["message"] = message
    completed = 0.0
    for step in project["pipeline"]["steps"]:
        p = 100 if step["state"] == "complete" else step.get("progress", 0)
        completed += p / len(project["pipeline"]["steps"])
    project["pipeline"]["progress"] = min(100, int(round(completed)))
    save_project(project)


def _validate_step_artifacts(project: dict, step_key: str) -> bool:
    """Validate that required artifacts for a step exist and are complete."""
    root = project_dir(project["id"])
    outputs = root / "outputs"
    validators = {
        "tracking": lambda: (
            (outputs / "tracking" / "tracking" / "tracking_mot.txt").is_file()
            and (outputs / "tracking" / "tracking" / "tracking_mot.txt").stat().st_size > 100
        ),
        "jersey": lambda: (
            (outputs / "number_ocr" / "jersey_number_results.csv").is_file()
            and (outputs / "team_hints.csv").is_file()
        ),
        "events": lambda: (
            (outputs / "match_analysis" / "metric_running" / "player_running_summary.csv").is_file()
            and (outputs / "match_analysis" / "analysis" / "pass_events.csv").is_file()
            and (outputs / "events_for_annotation.json").is_file()
        ),
        "report": lambda: (
            _player_cards_complete(outputs)
            and (outputs / "match_report.html").is_file()
        ),
    }
    validator = validators.get(step_key)
    if validator is None:
        return True
    try:
        return validator()
    except (OSError, ValueError):
        return False


def _clean_failed_step_outputs(outputs: Path, step_key: str) -> None:
    """Remove corrupted/partial outputs from a failed step to allow clean retry."""
    cleanup_targets = {
        "tracking": [
            outputs / "tracking",
            outputs / "identity_audit",
        ],
        "jersey": [
            outputs / "number_ocr",
            outputs / "team_hints.csv",
        ],
        "events": [
            outputs / "match_analysis",
            outputs / "events_for_annotation.json",
            outputs / "focus_events.json",
        ],
        "report": [
            outputs / "player_cards",
            outputs / "player_cards_formal",
            outputs / "highlights",
            outputs / "match_report.html",
            outputs / "metric_pitch_replay.mp4",
            outputs / "artifact_manifest.json",
        ],
    }
    for path in cleanup_targets.get(step_key, []):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)


def _reset_from(project: dict, from_step: str) -> None:
    if from_step not in _STAGE_ORDER:
        raise ValueError("无效的恢复步骤")
    idx = _STAGE_ORDER.index(from_step)
    root = project_dir(project["id"])
    outputs = root / "outputs"

    for i, step in enumerate(project["pipeline"]["steps"]):
        if i < idx:
            if _validate_step_artifacts(project, step["key"]):
                step["state"] = "complete"
                step["progress"] = 100
            else:
                step["state"] = "failed"
                step["progress"] = 0
                step["message"] = f"{step['key']} 步骤的结果文件不完整，无法跳过"
                raise ValueError(f"前置步骤 {step['key']} 的结果文件不完整，无法从 {from_step} 继续")
        else:
            step["state"] = "pending"
            step["progress"] = 0
            step["message"] = ""

    _clean_failed_step_outputs(outputs, from_step)

    project["pipeline"].update({
        "state": "running", "progress": int(idx / len(_STAGE_ORDER) * 100), "current_step": from_step,
        "message": "准备处理", "started_at": now_iso(), "finished_at": None, "error": None,
        "attempt": int(project["pipeline"].get("attempt", 0)) + 1, "resume_from": from_step,
    })
    project["status"] = "running"


def build_tracking_config(project: dict, path: Path) -> Path:
    s = project["settings"]
    video = project["video"]
    config = {
        "schema_version": 2,
        "venue": project["id"],
        "video": {"path": video["path"], "fps": video["fps"], "width": video["width"], "height": video["height"], "duration_seconds": video["duration_seconds"]},
        "scene": {"camera_motion": "pan_rotate", "expected_on_field_players": int(s["expected_players"]), "referee_present": True},
        "detector": {"weights": s["weights_path"], "confidence": float(s["confidence"]), "imgsz": int(s["imgsz"]), "person_class": 0},
        "tracker": {"config_file": str((ENGINE_ROOT / "tracking" / "config" / "botsort_buffer.yaml").resolve()), "vid_stride": 1},
        "field_filter": {
            "enabled": True, "min_turf_score": float(s["min_turf_score"]), "min_track_turf_ratio": float(s["min_track_turf_ratio"]),
            "min_foot_y_ratio": float(s["min_foot_y_ratio"]), "min_geometry_ratio": 0.0,
            "geometry": {"enabled": False, "mode": "disabled"},
        },
        # expected_players is a prior only; keep_all_clusters avoids forcing false merges.
        "identity": {
            "max_ids": int(s["expected_players"]), "keep_all_clusters": True,
            "min_track_frames": int(s["min_track_frames"]), "min_presence_ratio": float(s["min_presence_ratio"]),
            "team_clusters": int(s["team_clusters"]), "team_names": [], "team_prototypes": [],
        },
        "calibration": {"enabled": False, "mode": "disabled", "validated": False},
        "metric_motion": {"enabled": False},
        "events": {"percentile": float(s["event_percentile"]), "minimum_gap_seconds": float(s["event_min_gap"]), "edge_margin_seconds": 1.0},
        "highlights": {"event_count": int(s["event_count"]), "seconds_before_event": float(s["pre_sec"]), "seconds_after_event": float(s["post_sec"])},
        "review": {"multimodal_direct_scoring": False},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _artifact_manifest(project: dict, outputs: Path) -> Path:
    patterns = [
        ("tracking_mot", outputs / "tracking" / "tracking" / "tracking_mot.txt"),
        ("tracking_video", outputs / "tracking" / "tracking" / "tracking_vis.mp4"),
        ("identity_audit", outputs / "identity_audit" / "audit_report.json"),
        ("identity_audit_transitions", outputs / "identity_audit" / "transitions.csv"),
        ("jersey_results", outputs / "number_ocr" / "jersey_number_results.csv"),
        ("running_summary", outputs / "match_analysis" / "metric_running" / "player_running_summary.csv"),
        ("running_timeseries", outputs / "match_analysis" / "metric_running" / "player_running_timeseries.csv"),
        ("pass_events", outputs / "match_analysis" / "analysis" / "pass_events.csv"),
        ("possession_intervals", outputs / "match_analysis" / "analysis" / "possession_intervals.csv"),
        ("quality_report", outputs / "match_analysis" / "analysis" / "quality_report.json"),
        ("events", outputs / "events_for_annotation.json"),
        ("match_report", outputs / "match_report.html"),
        ("metric_pitch_video", outputs / "metric_pitch_replay.mp4"),
        ("focus_manifest", outputs / "highlights" / "id_focus_clips.json"),
        ("calibration", Path(project["calibration"]["path"])),
    ]
    artifacts = []
    for key, path in patterns:
        if path.is_file():
            artifacts.append({"key": key, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    # Player cards/highlights can contain many files; include counts and a few hashes rather than an enormous project.json.
    for folder_key, folder in [("player_cards", outputs / "player_cards"), ("formal_cards", outputs / "player_cards_formal"), ("highlights", outputs / "highlights")]:
        if folder.is_dir():
            files = sorted(p for p in folder.rglob("*") if p.is_file())
            artifacts.append({"key": folder_key, "path": str(folder.resolve()), "file_count": len(files), "total_size_bytes": sum(p.stat().st_size for p in files)})
    manifest = {
        "schema_version": 2,
        "project_id": project["id"],
        "generated_at": now_iso(),
        "video": {k: project["video"].get(k) for k in ("filename", "fps", "width", "height", "frame_count", "duration_seconds", "size_bytes")},
        "calibration_validation": project["calibration"].get("validation"),
        "artifacts": artifacts,
    }
    path = outputs / "artifact_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _select_focus_events(source: Path, dest: Path, limit: int, fps: float = 30.0) -> int:
    if not source.is_file():
        return 0
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    rows = payload.get("events", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("事件文件必须是事件数组，或包含 events 数组的对象")
    usable: list[dict] = []
    fps = max(0.001, float(fps))
    for source_index, source_row in enumerate(rows):
        if not isinstance(source_row, dict) or source_row.get("primary_global_id") is None:
            continue
        row = dict(source_row)
        start_frame = row.get("start_frame_proc")
        end_frame = row.get("end_frame_proc")
        event_frame = row.get("event_frame_proc")
        if start_frame is None or end_frame is None:
            if row.get("start_time") is None or row.get("end_time") is None:
                continue
            start_frame = round(float(row["start_time"]) * fps)
            end_frame = round(float(row["end_time"]) * fps)
        start_frame, end_frame = int(start_frame), int(end_frame)
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        if event_frame is None:
            anchor = row.get("anchor_time")
            event_frame = round(float(anchor) * fps) if anchor is not None else (start_frame + end_frame) // 2
        row.update({
            "source_event_id": source_row.get("event_id"),
            "event_id": source_index,
            "start_frame_proc": start_frame,
            "event_frame_proc": max(start_frame, min(int(event_frame), end_frame - 1)),
            "end_frame_proc": end_frame,
            "base_event_type": row.get("base_event_type") or row.get("event_type"),
            "actor_attribution_status": row.get("actor_attribution_status") or row.get("machine_status"),
        })
        usable.append(row)
    priority = {"goal_candidate": 0, "counterpress_recovery": 1, "shielding_under_pressure": 2}
    usable.sort(key=lambda r: (
        priority.get(str(r.get("base_event_type") or r.get("event_type") or ""), 9),
        -float(r.get("score") or r.get("confidence") or 0),
    ))
    # Round-robin by player so the product does not produce twelve clips of the same ID.
    buckets: dict[int, list[dict]] = {}
    for row in usable:
        buckets.setdefault(int(row["primary_global_id"]), []).append(row)
    selected: list[dict] = []
    while len(selected) < limit and any(buckets.values()):
        for gid in sorted(buckets):
            if buckets[gid] and len(selected) < limit:
                selected.append(buckets[gid].pop(0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(selected)



def _mot_global_ids(path: Path) -> list[int]:
    gids: set[int] = set()
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            values = line.split(",")
            if len(values) < 2:
                continue
            try:
                gids.add(int(float(values[1])))
            except (TypeError, ValueError):
                continue
    return sorted(gids)

def _internal_player_cards_complete(outputs: Path) -> bool:
    """The internal package is the stable source for resumable formal export."""
    return (
        (outputs / "player_cards" / "package_manifest.json").is_file()
        and (outputs / "player_cards" / "summary.txt").is_file()
    )


def _player_cards_complete(outputs: Path) -> bool:
    """The formal summary is written only after every player has been exported."""
    return _internal_player_cards_complete(outputs) and (
        outputs / "player_cards_formal" / "summary.txt"
    ).is_file()


def _video_output_complete(path: Path, *, minimum_bytes: int = 1024) -> bool:
    """Reject empty/truncated containers while allowing completed media to be reused."""
    try:
        return path.is_file() and path.stat().st_size > minimum_bytes
    except OSError:
        return False


def _highlights_complete(outputs: Path) -> bool:
    manifest = outputs / "highlights" / "id_focus_clips.json"
    try:
        rows = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(rows, list):
        return False
    return all(
        isinstance(row, dict)
        and bool(row.get("clip_file"))
        and _video_output_complete(manifest.parent / str(row["clip_file"]))
        for row in rows
    )


def _clear_downstream(
    outputs: Path,
    from_step: str,
    *,
    preserve_player_cards: bool = False,
    preserve_highlights: bool = False,
    preserve_metric_replay: bool = False,
) -> None:
    targets = {
        "tracking": [outputs],
        "jersey": [outputs / "number_ocr", outputs / "team_hints.csv", outputs / "match_analysis", outputs / "events_for_annotation.json", outputs / "player_cards", outputs / "player_cards_formal", outputs / "highlights", outputs / "match_report.html", outputs / "metric_pitch_replay.mp4", outputs / "artifact_manifest.json"],
        "events": [outputs / "match_analysis", outputs / "events_for_annotation.json", outputs / "player_cards", outputs / "player_cards_formal", outputs / "highlights", outputs / "match_report.html", outputs / "metric_pitch_replay.mp4", outputs / "artifact_manifest.json"],
        "report": [outputs / "player_cards", outputs / "player_cards_formal", outputs / "highlights", outputs / "match_report.html", outputs / "metric_pitch_replay.mp4", outputs / "artifact_manifest.json"],
    }
    preserved = {outputs / "player_cards", outputs / "player_cards_formal"} if preserve_player_cards else set()
    if preserve_highlights:
        preserved.add(outputs / "highlights")
    if preserve_metric_replay:
        preserved.add(outputs / "metric_pitch_replay.mp4")
    for path in targets[from_step]:
        if path in preserved:
            continue
        if path == outputs:
            if path.exists(): shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


class PipelineRunner:
    def __init__(self, project_id: str, from_step: str = "tracking"):
        self.project_id = project_id
        self.from_step = from_step
        self.cancelled = threading.Event()
        self.process: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.run_id = uuid.uuid4().hex[:12]

    def start(self) -> None:
        self.thread = threading.Thread(target=self.run, name=f"pipeline-{self.project_id}-{self.run_id}", daemon=True)
        self.thread.start()

    def cancel(self) -> None:
        self.cancelled.set()
        if self.process and self.process.poll() is None:
            try: self.process.terminate()
            except Exception: pass

    def _command(self, command: list[str], log_path: Path, progress_cb: Callable[[str], None] | None = None, env: dict | None = None) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        child_env = os.environ.copy()
        if env:
            child_env.update(env)
        # Python switches to block buffering when stdout is piped.  Force live
        # lines so long-running tracking stages can update the product UI.
        child_env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n$ " + " ".join(command) + "\n")
            log.flush()
            self.process = subprocess.Popen(command, cwd=SYSTEM_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, encoding="utf-8", errors="replace", bufsize=1, env=child_env)
            assert self.process.stdout is not None
            for line in self.process.stdout:
                log.write(line); log.flush()
                if progress_cb: progress_cb(line)
                if self.cancelled.is_set():
                    self.process.terminate(); raise RuntimeError("任务已取消")
            code = self.process.wait()
            self.process = None
            if code != 0:
                raise subprocess.CalledProcessError(code, command)

    def _record_start(self, project: dict) -> None:
        project["run_history"].append({
            "run_id": self.run_id, "attempt": project["pipeline"]["attempt"], "started_at": now_iso(), "finished_at": None,
            "from_step": self.from_step, "state": "running", "error": None,
        })
        project["run_history"] = project["run_history"][-20:]
        save_project(project)

    def _record_finish(self, project: dict, state: str, error: str | None = None) -> None:
        for row in reversed(project.get("run_history", [])):
            if row.get("run_id") == self.run_id:
                row.update(state=state, finished_at=now_iso(), error=error)
                break

    def run(self) -> None:
        project = load_project(self.project_id)
        root = project_dir(self.project_id)
        outputs = root / "outputs"
        log = root / "logs" / f"pipeline_{self.run_id}.log"
        latest_log = root / "logs" / "pipeline.log"
        try:
            if not project.get("video"):
                raise ValueError("请先上传视频")
            if project.get("calibration", {}).get("status") != "ready":
                raise ValueError("请先完成并通过动态标定")
            weights = Path(project["settings"]["weights_path"])
            if not weights.is_file():
                raise FileNotFoundError(f"分析模型不存在：{weights}")
            reuse_player_cards = self.from_step == "report" and _internal_player_cards_complete(outputs)
            formal_cards_complete = _player_cards_complete(outputs)
            reuse_highlights = self.from_step == "report" and _highlights_complete(outputs)
            reuse_metric_replay = self.from_step == "report" and _video_output_complete(outputs / "metric_pitch_replay.mp4")
            _clear_downstream(
                outputs,
                self.from_step,
                preserve_player_cards=reuse_player_cards,
                preserve_highlights=reuse_highlights,
                preserve_metric_replay=reuse_metric_replay,
            )
            _reset_from(project, self.from_step)
            self._record_start(project)
            start_idx = _STAGE_ORDER.index(self.from_step)

            tracking_out = outputs / "tracking"
            mot = tracking_out / "tracking" / "tracking_mot.txt"
            number_out = outputs / "number_ocr"
            team_hints = outputs / "team_hints.csv"
            analysis_out = outputs / "match_analysis"
            events_json = outputs / "events_for_annotation.json"
            cards = outputs / "player_cards"
            formal = outputs / "player_cards_formal"

            # 1) Tracking
            if start_idx <= 0:
                _clean_failed_step_outputs(outputs, "tracking")
                _set_step(project, "tracking", state="running", progress=2, message="正在追踪球员与足球…")
                cfg = build_tracking_config(project, root / "config" / "tracking.yaml")
                def track_progress(line: str):
                    update = _tracking_progress(line, int(project["video"]["frame_count"]))
                    if update:
                        progress, message = update
                        _set_step(project, "tracking", state="running", progress=progress, message=message)
                self._command([sys.executable, str(ENGINE_ROOT / "tracking" / "run_pipeline.py"), "--config", str(cfg), "--output", str(tracking_out),
                               "--device", str(project["settings"]["device"]), "--vid-stride", "1"], log, track_progress)
                if not mot.is_file(): raise FileNotFoundError("追踪结果未生成")
                if bool(project["settings"].get("identity_audit_enabled", True)):
                    _set_step(project, "tracking", state="running", progress=96, message="正在进行身份质量审计…")
                    audit_out = outputs / "identity_audit"
                    if audit_out.exists():
                        shutil.rmtree(audit_out)
                    gids = _mot_global_ids(mot)
                    if gids:
                        audit_env = os.environ.copy()
                        audit_env["PYTHONPATH"] = os.pathsep.join([
                            str(ENGINE_ROOT / "tracking"),
                            str(ENGINE_ROOT / "identity_audit"),
                            audit_env.get("PYTHONPATH", ""),
                        ])
                        audit_cmd = [
                            sys.executable, str(ENGINE_ROOT / "identity_audit" / "mode_split" / "audit_mot.py"),
                            "--video", project["video"]["path"], "--mot", str(mot), "--output", str(audit_out),
                            "--clusters", str(max(2, min(3, int(project["settings"].get("team_clusters", 3))))),
                            "--sample-stride-frames", str(int(project["settings"].get("identity_audit_sample_stride", 30))),
                            "--only-gids", *[str(g) for g in gids],
                        ]
                        try:
                            self._command(audit_cmd, log, env=audit_env)
                        except Exception as audit_exc:
                            audit_out.mkdir(parents=True, exist_ok=True)
                            (audit_out / "audit_error.json").write_text(
                                json.dumps({"status": "warning", "message": str(audit_exc), "mot_was_modified": False}, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                            with log.open("a", encoding="utf-8") as handle:
                                handle.write(f"\n[identity-audit-warning] {audit_exc}\n")
                _set_step(project, "tracking", state="complete", progress=100, message="追踪完成")
            elif not mot.is_file():
                raise FileNotFoundError("无法从号码识别继续：缺少追踪结果")

            # 2) Jersey number + team grouping
            if start_idx <= 1:
                if start_idx > 0 and not mot.is_file():
                    raise FileNotFoundError("无法从号码识别继续：缺少追踪结果文件 tracking_mot.txt")
                _clean_failed_step_outputs(outputs, "jersey")
                _set_step(project, "jersey", state="running", progress=5, message="正在识别球衣号码…")
                self._command([sys.executable, "-m", "app.services.team_hints", "--video", project["video"]["path"], "--mot", str(mot),
                               "--output", str(team_hints), "--clusters", str(project["settings"]["team_clusters"]), "--samples", str(project["settings"]["team_samples_per_id"])], log)
                _set_step(project, "jersey", state="running", progress=25, message="正在聚合多帧号码证据…")
                ocr_cmd = [sys.executable, str(ENGINE_ROOT / "match_analysis" / "run_jersey_ocr.py"), "--video", project["video"]["path"], "--mot", str(mot),
                           "--team-hints", str(team_hints), "--output", str(number_out), "--maximum-candidates-per-id", str(project["settings"]["ocr_candidates_per_id"])]
                if str(project["settings"]["device"]).lower() == "cpu": ocr_cmd.append("--cpu")
                self._command(ocr_cmd, log)
                if not (number_out / "jersey_number_results.csv").is_file(): raise FileNotFoundError("号码识别结果未生成")
                _set_step(project, "jersey", state="complete", progress=100, message="号码识别完成")
            elif not team_hints.is_file():
                raise FileNotFoundError("无法从事件检测继续：缺少球队分组结果")

            # 3) Metric running + possession/pass/event
            if start_idx <= 2:
                if start_idx > 0 and not team_hints.is_file():
                    raise FileNotFoundError("无法从事件检测继续：缺少球队分组结果文件 team_hints.csv")
                if start_idx > 0 and not mot.is_file():
                    raise FileNotFoundError("无法从事件检测继续：缺少追踪结果文件 tracking_mot.txt")
                _clean_failed_step_outputs(outputs, "events")
                _set_step(project, "events", state="running", progress=5, message="正在计算米制跑动、球权与传球…")
                env = os.environ.copy(); src = ENGINE_ROOT / "football_metric_running" / "src"
                env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
                self._command([sys.executable, str(ENGINE_ROOT / "match_analysis" / "run_integrated_analysis.py"), "--tracking-dir", str(tracking_out),
                               "--calibration", project["calibration"]["path"], "--video", project["video"]["path"], "--output", str(analysis_out),
                               "--team-map", str(team_hints), "--running-src", str(src), "--team-clusters", str(project["settings"]["team_clusters"]),
                               "--min-pass-displacement-m", str(project["settings"]["min_pass_displacement_m"])], log, env=env)
                _set_step(project, "events", state="running", progress=78, message="正在整理比赛事件…")
                self._command([sys.executable, str(ENGINE_ROOT / "match_analysis" / "build_machine_events.py"), "--video", project["video"]["path"], "--mot", str(mot),
                               "--ball", str(tracking_out / "tracking" / "ball_positions_observed.csv"), "--stage4-events", str(tracking_out / "tracking" / "events.json"),
                               "--analysis-dir", str(analysis_out / "analysis"), "--output", str(events_json), "--stage4-sample-size", str(project["settings"]["event_count"]),
                               "--field-length-m", str(project["settings"]["field_length_m"]), "--field-width-m", str(project["settings"]["field_width_m"])], log)
                if not (analysis_out / "metric_running" / "player_running_summary.csv").is_file(): raise FileNotFoundError("米制跑动结果未生成")
                if not (analysis_out / "analysis" / "pass_events.csv").is_file(): raise FileNotFoundError("传球结果未生成")
                _set_step(project, "events", state="complete", progress=100, message="事件检测完成")
            elif not events_json.is_file():
                raise FileNotFoundError("无法从报告生成继续：缺少事件结果")

            # 4) Full product deliverables
            if start_idx <= 2:
                if not events_json.is_file():
                    raise FileNotFoundError("无法从报告生成继续：缺少事件结果文件 events_for_annotation.json")
                if not mot.is_file():
                    raise FileNotFoundError("无法从报告生成继续：缺少追踪结果文件 tracking_mot.txt")
            _clean_failed_step_outputs(outputs, "report")
            _set_step(
                project, "report", state="running", progress=40 if reuse_player_cards else 5,
                message="复用已完成的球员卡，继续生成高光…" if reuse_player_cards else "正在生成球员卡…",
            )
            timeseries = analysis_out / "metric_running" / "player_running_timeseries.csv"
            def card_progress(line: str):
                update = _player_card_progress(line)
                if update:
                    progress, message = update
                    _set_step(project, "report", state="running", progress=progress, message=message)
            if not reuse_player_cards:
                self._command([sys.executable, str(ENGINE_ROOT / "match_analysis" / "generate_player_card.py"), "--video", project["video"]["path"],
                               "--mot", str(mot), "--numbers", str(number_out / "clip_eligibility.json"), "--events", str(events_json), "--calibration", project["calibration"]["path"],
                               "--running-timeseries", str(timeseries), "--fps", str(project["video"]["fps"]), "--output", str(cards), "--formal-output", str(formal)], log, card_progress)
            elif not formal_cards_complete:
                self._command([
                    sys.executable, str(ENGINE_ROOT / "match_analysis" / "export_player_card_delivery_v1.py"),
                    "--internal-package", str(cards), "--output", str(formal), "--resume",
                    "--running-timeseries", str(timeseries),
                ], log, card_progress)

            _set_step(project, "report", state="running", progress=42, message="正在生成 TARGET 高光…")
            focus_source = outputs / "focus_events.json"
            selected = _select_focus_events(
                events_json, focus_source,
                int(project["settings"].get("focus_clip_limit", 12)),
                float(project["video"]["fps"]),
            )
            highlights_dir = outputs / "highlights"
            if selected and not reuse_highlights:
                self._command([sys.executable, str(ENGINE_ROOT / "tracking" / "make_id_focus_clips.py"), "--video", project["video"]["path"], "--mot", str(mot),
                               "--events", str(focus_source), "--outdir", str(highlights_dir), "--manifest", str(highlights_dir / "id_focus_clips.json"),
                               "--vid-stride", "1", "--include-review"], log)
            elif not selected and not reuse_highlights:
                highlights_dir.mkdir(parents=True, exist_ok=True)
                (highlights_dir / "id_focus_clips.json").write_text("[]", encoding="utf-8")

            _set_step(project, "report", state="running", progress=62, message="正在生成 2D 赛后回放…")
            replay_mp4 = outputs / "metric_pitch_replay.mp4"
            if timeseries.is_file() and not reuse_metric_replay:
                self._command([sys.executable, str(ENGINE_ROOT / "match_analysis" / "render_metric_pitch.py"), "--calibration", project["calibration"]["path"],
                               "--timeseries", str(timeseries), "--analysis-dir", str(analysis_out / "analysis"), "--output", str(replay_mp4)], log)

            _set_step(project, "report", state="running", progress=84, message="正在排版比赛报告并归档结果…")
            from app.services.reporting import build_match_report
            build_match_report(project, outputs)
            manifest_path = _artifact_manifest(project, outputs)

            project = load_project(self.project_id)
            project["pipeline"].update(state="complete", progress=100, message="分析完成", finished_at=now_iso(), current_step="report", error=None)
            for step in project["pipeline"]["steps"]:
                step.update(state="complete", progress=100, message="完成")
            project["status"] = "complete"
            project["outputs"] = {
                "tracking": str(tracking_out), "match_analysis": str(analysis_out), "number_ocr": str(number_out),
                "player_cards": str(cards), "formal_cards": str(formal), "highlights": str(highlights_dir),
                "metric_pitch_video": str(replay_mp4), "report_html": str(outputs / "match_report.html"), "artifact_manifest": str(manifest_path),
            }
            project["artifact_manifest"] = str(manifest_path)
            self._record_finish(project, "complete")
            save_project(project)
        except Exception as exc:
            project = load_project(self.project_id)
            cancelled = self.cancelled.is_set() or str(exc) == "任务已取消"
            state = "cancelled" if cancelled else "failed"
            cur = project["pipeline"].get("current_step")
            failed_step = cur
            if not cancelled and cur:
                for step in project["pipeline"]["steps"]:
                    if step["key"] == cur and step["state"] == "running":
                        step["state"] = "failed"
                        step["message"] = str(exc)
                        failed_step = step["key"]
                        break
                else:
                    for step in project["pipeline"]["steps"]:
                        if step["state"] == "running":
                            step["state"] = "failed"
                            step["message"] = str(exc)
                            failed_step = step["key"]
                            break
            if cancelled:
                for step in project["pipeline"]["steps"]:
                    if step["state"] == "running":
                        step["state"] = "pending"
                        step["message"] = ""
                        break
            retry_hint = f"处理失败，可从 {failed_step} 步骤继续" if not cancelled else "任务已取消，可重新运行"
            project["pipeline"].update(state=state, error=str(exc), message=retry_hint, finished_at=now_iso())
            project["status"] = state
            self._record_finish(project, state, str(exc))
            save_project(project)
        finally:
            # Keep a stable latest log for support UI without losing per-attempt history.
            try:
                if log.is_file(): shutil.copy2(log, latest_log)
            except Exception: pass
            with _LOCK:
                _RUNNERS.pop(self.project_id, None)


def start_pipeline(project_id: str, from_step: str = "tracking") -> None:
    if from_step not in _STAGE_ORDER:
        raise ValueError("无效的处理步骤")
    project = load_project(project_id)
    pipeline_state = project.get("pipeline", {}).get("state")
    if pipeline_state == "running":
        raise RuntimeError("该项目已有任务正在处理，请勿重复启动")
    with _LOCK:
        if project_id in _RUNNERS:
            runner = _RUNNERS[project_id]
            if runner.thread and runner.thread.is_alive():
                raise RuntimeError("该项目正在处理")
            else:
                _RUNNERS.pop(project_id, None)
        if len(_RUNNERS) >= _MAX_CONCURRENT_PIPELINES:
            raise RuntimeError(f"分析资源正在被其他比赛占用；当前最多同时运行 {_MAX_CONCURRENT_PIPELINES} 个正式任务")
        runner = PipelineRunner(project_id, from_step=from_step)
        _RUNNERS[project_id] = runner
        runner.start()


def cancel_pipeline(project_id: str) -> None:
    with _LOCK:
        runner = _RUNNERS.get(project_id)
    if runner: runner.cancel()


def expand_dynamic_calibration(project_id: str) -> None:
    project = load_project(project_id)
    anchors = project.get("calibration", {}).get("anchors") or []
    paths = [Path(str(row.get("path") or "")) for row in anchors if row.get("passed")]
    if not paths:
        # Backward compatibility with V1 single reference.
        ref = Path(project["calibration"].get("reference_path") or "")
        if ref.is_file(): paths = [ref]
    if not paths or any(not p.is_file() for p in paths):
        raise FileNotFoundError("缺少通过验证的标定锚点")
    root = project_dir(project_id)
    out = root / "calibration" / "dynamic_calibration.json"
    log = root / "logs" / "calibration.log"
    if out.exists(): out.unlink()
    src = ENGINE_ROOT / "football_metric_running" / "src"
    env = os.environ.copy(); env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "running_metrics_v1.build_multi_anchor_dynamic_calibration"]
    for path in paths:
        cmd += ["--anchor", str(path)]
    cmd += ["--output", str(out), "--sample-step", str(project["settings"]["dynamic_sample_step"]), "--max-interpolation-gap", str(project["settings"]["dynamic_max_gap"])]
    with log.open("a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + "\n"); f.flush()
        subprocess.run(cmd, cwd=SYSTEM_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    summary = summarize_dynamic(out)
    anchors_valid = all(bool(json.loads(p.read_text(encoding="utf-8")).get("validation", {}).get("passed", False)) for p in paths)
    coverage = float(summary.get("accepted_ratio") or 0)
    threshold = float(project["settings"].get("dynamic_min_coverage", 0.8))
    passed = anchors_valid and coverage >= threshold
    dynamic = json.loads(out.read_text(encoding="utf-8"))
    dynamic.setdefault("validation", {}).update({
        "passed": passed, "all_anchor_scale_validations_passed": anchors_valid,
        "accepted_ratio": coverage, "minimum_accepted_ratio": threshold,
    })
    out.write_text(json.dumps(dynamic, ensure_ascii=False), encoding="utf-8")
    project = load_project(project_id)
    project["calibration"].update(
        status="ready" if passed else "failed", source="manual_multi_anchor_auto_rotation", path=str(out),
        validation={**summary, "passed": passed, "minimum_accepted_ratio": threshold},
        message=(f"动态标定已生成：{len(paths)} 个视角锚点，有效覆盖 {coverage:.1%}" if passed
                 else f"动态标定未通过：有效覆盖 {coverage:.1%}，要求至少 {threshold:.0%}"),
    )
    save_project(project)


def start_dynamic_calibration(project_id: str) -> None:
    with _LOCK:
        if project_id in _CALIBRATORS and _CALIBRATORS[project_id].is_alive():
            raise RuntimeError("该项目正在生成动态标定")
    project = load_project(project_id)
    anchors = [a for a in project.get("calibration", {}).get("anchors", []) if a.get("passed")]
    legacy_ready = project.get("calibration", {}).get("status") == "reference_ready" and project.get("calibration", {}).get("reference_path")
    if not anchors and not legacy_ready:
        raise ValueError("请先添加至少 1 个通过验证的动态标定锚点")
    project["calibration"]["status"] = "building"
    project["calibration"]["message"] = f"正在用 {max(1, len(anchors))} 个视角锚点生成全片逐帧动态标定…"
    save_project(project)

    def job() -> None:
        try:
            expand_dynamic_calibration(project_id)
        except Exception as exc:
            p = load_project(project_id)
            p["calibration"].update(status="failed", message=f"动态标定生成失败：{exc}")
            save_project(p)
        finally:
            with _LOCK: _CALIBRATORS.pop(project_id, None)

    thread = threading.Thread(target=job, name=f"calibration-{project_id}", daemon=True)
    with _LOCK: _CALIBRATORS[project_id] = thread
    thread.start()
