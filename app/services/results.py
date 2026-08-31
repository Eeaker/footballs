from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import DEMO_ROOT
from app.services.storage import project_dir


@lru_cache(maxsize=48)
def _read_csv_snapshot(path_text: str, modified_ns: int, size_bytes: int) -> tuple[dict[str, str], ...]:
    del modified_ns, size_bytes  # values intentionally participate in the cache key
    with Path(path_text).open("r", encoding="utf-8-sig", newline="") as f:
        return tuple(csv.DictReader(f))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    stat = path.stat()
    # Return a shallow list copy so callers can sort/filter without modifying
    # the shared immutable snapshot. Large analysis CSVs are parsed only once.
    return list(_read_csv_snapshot(str(path.resolve()), stat.st_mtime_ns, stat.st_size))


def _read_json(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try: return int(float(value))
    except Exception: return default


def _first_present(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first non-empty value without treating numeric zero as missing."""
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def output_paths(project: dict[str, Any]) -> dict[str, Path]:
    if project.get("kind") == "demo":
        base = DEMO_ROOT
        return {
            "root": base,
            "tracking": base / "tracking",
            "analysis": base / "match_analysis" / "analysis",
            "running": base / "match_analysis" / "metric_running",
            "ocr": base / "number_ocr",
            "cards": base / "player_cards",
            "formal_cards": base / "player_cards",
            "highlights": base / "highlights",
            "events": base / "events_for_annotation.json",
            "report_html": base / "nati_report" / "report.html",
            "report_pdf": base / "nati_report" / "report.pdf",
            "demo_video": base / "source_preview.mp4",
            "replay_video": base / "metric_pitch_replay.mp4",
            "identity_audit": base / "identity_audit",
        }
    root = project_dir(project["id"]) / "outputs"
    return {
        "root": root,
        "tracking": root / "tracking",
        "analysis": root / "match_analysis" / "analysis",
        "running": root / "match_analysis" / "metric_running",
        "ocr": root / "number_ocr",
        "cards": root / "player_cards",
        "formal_cards": root / "player_cards_formal",
        "highlights": root / "highlights",
        "events": root / "events_for_annotation.json",
        "report_html": root / "match_report.html",
        "report_pdf": root / "match_report.pdf",
        "replay_video": root / "metric_pitch_replay.mp4",
        "artifact_manifest": root / "artifact_manifest.json",
        "identity_audit": root / "identity_audit",
    }


def _team_label(project: dict[str, Any], team_id: str | None) -> str:
    if not team_id:
        return "未分组"
    custom = (project.get("match") or {}).get("team_labels") or {}
    if team_id in custom and custom[team_id]: return str(custom[team_id])
    if team_id == "team_0": return "队伍 A"
    if team_id == "team_1": return "队伍 B"
    if team_id == "team_2": return "裁判/其他"
    return str(team_id)


@lru_cache(maxsize=12)
def _semantic_event_snapshot(
    running_path: str, running_stamp: int, team_path: str, team_stamp: int,
    possession_path: str, possession_stamp: int, transition_path: str, transition_stamp: int,
    evidence_path: str, evidence_stamp: int, stage4_path: str, stage4_stamp: int,
    fps: float, duration: float, field_length: float, field_width: float,
) -> tuple[dict[str, Any], ...]:
    del running_stamp, team_stamp, possession_stamp, transition_stamp, evidence_stamp, stage4_stamp
    from engine.match_analysis.analysis_lib.semantic_events import derive_semantic_events

    running = _read_csv(Path(running_path)); teams = _read_csv(Path(team_path))
    positions: dict[tuple[int, int], tuple[float, float]] = {}
    for row in running:
        gid = _int(row.get("global_id"), -1); frame = _int(row.get("proc_idx"), -1)
        x = _float(_first_present(row, "x_m_smooth", "x_m_raw"), float("nan")); y = _float(_first_present(row, "y_m_smooth", "y_m_raw"), float("nan"))
        if gid >= 0 and frame >= 0 and math.isfinite(x) and math.isfinite(y): positions[(frame, gid)] = (x, y)
    team_map = {_int(row.get("global_id"), -1): str(row.get("team_id") or "") for row in teams}
    ball_metric: dict[int, tuple[float, float]] = {}
    for row in _read_csv(Path(evidence_path)):
        frame = _int(row.get("frame_proc"), -1)
        x = _float(row.get("ball_x_m"), float("nan")); y = _float(row.get("ball_y_m"), float("nan"))
        if frame >= 0 and math.isfinite(x) and math.isfinite(y): ball_metric[frame] = (x, y)
    stage4_payload = _read_json(Path(stage4_path), [])
    stage4 = stage4_payload.get("events", []) if isinstance(stage4_payload, dict) else stage4_payload
    return tuple(derive_semantic_events(
        fps=fps, duration_seconds=duration, field_length_m=field_length, field_width_m=field_width,
        team_map=team_map, positions=positions,
        possessions=_read_csv(Path(possession_path)), transitions=_read_csv(Path(transition_path)),
        stage4_events=stage4 or [], ball_metric_by_frame=ball_metric,
    ))


def semantic_events(project: dict[str, Any]) -> list[dict[str, Any]]:
    paths = output_paths(project); saved = _read_json(paths["analysis"] / "semantic_events.json", None)
    if isinstance(saved, dict) and isinstance(saved.get("events"), list): return saved["events"]
    files = [
        paths["running"] / "player_running_timeseries.csv", paths["analysis"] / "player_team_map.csv",
        paths["analysis"] / "possession_intervals.csv", paths["analysis"] / "possession_transitions.csv",
        paths["analysis"] / "possession_frame_evidence.csv",
    ]
    stage4 = paths["tracking"] / "tracking" / "events.json"
    if not stage4.is_file(): stage4 = paths["tracking"] / "events.json"
    if not all(path.is_file() for path in files) or not stage4.is_file(): return []
    stamps = [path.stat().st_mtime_ns for path in files] + [stage4.stat().st_mtime_ns]
    video = project.get("video") or {}; settings = project.get("settings") or {}
    return list(_semantic_event_snapshot(
        str(files[0]), stamps[0], str(files[1]), stamps[1], str(files[2]), stamps[2],
        str(files[3]), stamps[3], str(files[4]), stamps[4], str(stage4), stamps[5],
        _float(video.get("fps"), 30.0), _float(video.get("duration_seconds")),
        _float(settings.get("field_length_m"), 45.0), _float(settings.get("field_width_m"), 25.0),
    ))


def _tracking_count(paths: dict[str, Path]) -> int:
    candidates = [paths["tracking"] / "tracking" / "global_id_summary.json", paths["tracking"] / "global_id_summary.json"]
    for p in candidates:
        data = _read_json(p)
        if isinstance(data, list): return len(data)
        if isinstance(data, dict):
            for key in ("global_ids", "ids", "tracks"):
                if isinstance(data.get(key), list): return len(data[key])
            if "count" in data:
                try: return int(data["count"])
                except Exception: pass
    rows = _read_csv(paths["analysis"] / "player_team_map.csv")
    return len(rows)


def summary(project: dict[str, Any]) -> dict[str, Any]:
    paths = output_paths(project)
    running = _read_csv(paths["running"] / "player_running_summary.csv")
    passes = _read_csv(paths["analysis"] / "pass_events.csv")
    possessions = _read_csv(paths["analysis"] / "possession_intervals.csv")
    teams = _read_csv(paths["analysis"] / "team_pass_summary.csv")
    card_rows = _read_csv(paths["cards"] / "player_running_summary.csv")
    quality = _read_json(paths["analysis"] / "quality_report.json", {}) or {}
    numbers = _read_csv(paths["ocr"] / "jersey_number_results.csv")
    confirmed_numbers = sum(1 for row in numbers if "confirm" in str(row.get("status") or "").lower() or str(row.get("predicted_number") or row.get("final_number") or "").strip())
    peak = max((_float(row.get("peak_speed_mps_p95") or row.get("max_speed_mps")) for row in running), default=0)
    total_distance = sum(_float(row.get("total_distance_m") or row.get("total_distance")) for row in running)
    team_rows = [{**row, "label": _team_label(project, row.get("team_id"))} for row in teams]
    cal = project.get("calibration") or {}
    return {
        "project_id": project["id"], "name": project["name"], "status": project.get("status"), "kind": project.get("kind"),
        "match": project.get("match") or {},
        "candidate_ids": _tracking_count(paths), "confirmed_player_cards": len(card_rows),
        "total_distance_m": round(total_distance, 1), "peak_speed_mps": round(peak, 2),
        "pass_candidates": len(passes), "possession_intervals": len(possessions), "confirmed_numbers": confirmed_numbers,
        "teams": team_rows,
        "calibration": {"status": cal.get("status"), "validation": cal.get("validation"), "anchor_count": len(cal.get("anchors") or [])},
        "quality_status": quality.get("status") or (quality.get("overall", {}) or {}).get("status") or "available",
        "human_review_pending": bool(quality.get("status") == "pending_human_review"),
        "result_sections": {
            "replay": bool(_read_csv(paths["running"] / "player_running_timeseries.csv") or _read_csv(paths["analysis"] / "possession_frame_evidence.csv")),
            "highlights": paths["highlights"].is_dir(), "players": bool(card_rows or running), "report": paths["report_html"].is_file(), "replay_video": paths["replay_video"].is_file(),
        },
    }


def team_overview(project: dict[str, Any]) -> list[dict[str, Any]]:
    paths = output_paths(project)
    team_map = {r.get("team_id"): [] for r in _read_csv(paths["analysis"] / "team_pass_summary.csv")}
    for row in _read_csv(paths["analysis"] / "player_team_map.csv"):
        if row.get("team_id") in team_map:
            team_map[row["team_id"]].append(_int(row.get("global_id"), -1))
    running_by_id = {_int(r.get("global_id"), -1): r for r in _read_csv(paths["running"] / "player_running_summary.csv")}
    pass_summary = {r.get("team_id"): r for r in _read_csv(paths["analysis"] / "team_pass_summary.csv")}
    possessions = _read_csv(paths["analysis"] / "possession_intervals.csv")
    possession_frames = Counter()
    for r in possessions:
        possession_frames[r.get("team_id")] += max(0, _int(r.get("end_frame_proc")) - _int(r.get("start_frame_proc")) + 1)
    total_pos = sum(possession_frames.values())
    out = []
    for team_id in sorted(set(team_map) | set(pass_summary)):
        gids = team_map.get(team_id, [])
        distance = sum(_float(running_by_id.get(gid, {}).get("total_distance_m")) for gid in gids)
        peak = max((_float(running_by_id.get(gid, {}).get("peak_speed_mps_p95")) for gid in gids), default=0)
        p = pass_summary.get(team_id, {})
        out.append({
            "team_id": team_id, "label": _team_label(project, team_id), "player_ids": gids, "player_count": len(gids),
            "distance_m": round(distance, 1), "peak_speed_mps": round(peak, 2),
            "passes": _int(p.get("active_directed_passes")), "pass_distance_m": round(_float(p.get("total_pass_distance_m")), 1),
            "mean_pass_distance_m": round(_float(p.get("mean_pass_distance_m")), 2),
            "possession_share": round(possession_frames.get(team_id, 0) / total_pos, 4) if total_pos else None,
        })
    return out


def players(project: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the full technical-player set, enriched by curated player-card rows and human identity mappings.

    Formal output must not disappear just because only a subset of IDs has a curated card.
    """
    paths = output_paths(project)
    cards = _read_csv(paths["cards"] / "player_running_summary.csv")
    running = _read_csv(paths["running"] / "player_running_summary.csv")
    team_rows = {_int(r.get("global_id"), -1): r.get("team_id", "") for r in _read_csv(paths["analysis"] / "player_team_map.csv")}
    number_rows = {_int(r.get("global_id"), -1): r for r in _read_csv(paths["ocr"] / "jersey_number_results.csv")}
    identity_overrides = {}
    assessment_overrides = {}
    if project.get("kind") != "demo":
        try:
            from app.services.reviews import identity_mapping_dict, player_assessment_dict
            identity_overrides = identity_mapping_dict(project)
            assessment_overrides = player_assessment_dict(project)
        except Exception:
            identity_overrides = {}
            assessment_overrides = {}

    def apply_reviews(item: dict[str, Any]) -> dict[str, Any]:
        gids = item.get("global_ids") or []
        out = dict(item)
        mapping = next((identity_overrides.get(str(g)) for g in gids if identity_overrides.get(str(g))), None)
        if mapping:
            if mapping.get("name"):
                out["algorithm_player_id"] = item.get("player_id")
                out["player_id"] = mapping["name"]
                out["identity_status"] = "human_confirmed"
            if mapping.get("jersey_number"):
                out["jersey_number"] = mapping["jersey_number"]
            if mapping.get("team_id"):
                out["team_id"] = mapping["team_id"]
                out["team"] = _team_label(project, mapping["team_id"])
            out["identity_note"] = mapping.get("note") or ""
            person_key = str(mapping.get("person_key") or "").strip()
            if not person_key and mapping.get("roster_index") is not None:
                person_key = f"roster:{mapping['roster_index']}"
            if not person_key and mapping.get("name"):
                person_key = "manual:" + "|".join(str(mapping.get(k) or "").strip().casefold() for k in ("team_id", "jersey_number", "name"))
            out["_person_key"] = person_key or None
        assessment = next((assessment_overrides.get(str(g)) for g in gids if assessment_overrides.get(str(g))), None)
        out["assessment"] = assessment or {"scores": {}, "status": "pending", "note": "", "source": "human"}
        return out

    # Curated cards can consolidate one or more metric IDs. Index those IDs first.
    card_items: list[dict[str, Any]] = []
    card_gid_set: set[int] = set()
    for row in cards:
        gids = [_int(x, -1) for x in str(row.get("metric_global_ids") or "").replace(";", ",").split(",") if x.strip()]
        gids = [g for g in gids if g >= 0]
        card_gid_set.update(gids)
        team = row.get("team") or (team_rows.get(gids[0]) if gids else "")
        card_items.append({
            "player_id": row.get("player_id") or (f"ID {gids[0]}" if gids else "球员"),
            "jersey_number": row.get("jersey_number") or "—", "team_id": team, "team": _team_label(project, team), "global_ids": gids,
            "total_distance_m": _float(row.get("total_distance")), "sprint_count": _int(row.get("sprint_count")),
            "max_speed_mps": _float(row.get("max_speed_mps")), "speed_p95_mps": _float(row.get("speed_p95_mps")),
            "visible_time_sec": _float(row.get("tracked_visible_time_sec")), "identity_status": row.get("identity_resolution_status") or "candidate",
            "quality": row.get("data_quality") or "", "heatmap_path": row.get("heatmap_data_path") or None, "card_available": True,
        })

    out = [apply_reviews(r) for r in card_items]
    for row in running[:120]:
        gid = _int(row.get("global_id"), -1)
        if gid < 0 or gid in card_gid_set:
            continue
        team = team_rows.get(gid, ""); num = number_rows.get(gid, {})
        jersey = num.get("predicted_number") if "confirm" in str(num.get("status") or "").lower() else "待确认"
        item = {
            "player_id": f"ID {gid}", "jersey_number": jersey or "待确认", "team_id": team, "team": _team_label(project, team), "global_ids": [gid],
            "total_distance_m": _float(row.get("total_distance_m")), "sprint_count": _int(row.get("sprint_count")),
            "max_speed_mps": _float(row.get("peak_speed_mps_p95") or row.get("max_speed_mps")),
            "speed_p95_mps": _float(row.get("peak_speed_mps_p95")), "visible_time_sec": _float(row.get("valid_duration_sec")),
            "identity_status": "candidate", "quality": row.get("quality_flags") or "", "heatmap_path": None, "card_available": False,
        }
        out.append(apply_reviews(item))
    # Multiple fragmented technical IDs may be confirmed as one real player.
    # Consolidation happens only after an explicit human mapping gives them the
    # same person key; untouched candidate IDs always remain separate.
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    singles: list[dict[str, Any]] = []
    for item in out:
        key = item.get("_person_key")
        if key:
            grouped[str(key)].append(item)
        else:
            singles.append(item)
    merged = list(singles)
    for key, items in grouped.items():
        base = dict(items[0])
        gids = sorted({gid for item in items for gid in (item.get("global_ids") or [])})
        base["global_ids"] = gids
        base["technical_id_count"] = len(gids)
        base["total_distance_m"] = round(sum(_float(x.get("total_distance_m")) for x in items), 3)
        base["visible_time_sec"] = round(sum(_float(x.get("visible_time_sec")) for x in items), 3)
        base["sprint_count"] = sum(_int(x.get("sprint_count")) for x in items)
        base["max_speed_mps"] = max((_float(x.get("max_speed_mps")) for x in items), default=0.0)
        base["speed_p95_mps"] = max((_float(x.get("speed_p95_mps")) for x in items), default=0.0)
        base["card_available"] = any(bool(x.get("card_available")) for x in items)
        base["identity_status"] = "human_confirmed"
        base["person_key"] = key
        assessments = [x.get("assessment") or {} for x in items]
        confirmed = [(x.get("assessment") or {}, max(1.0, _float(x.get("visible_time_sec")))) for x in items if (x.get("assessment") or {}).get("status") == "confirmed"]
        if confirmed:
            score_keys = ("speed", "endurance", "running", "passing", "control", "shooting", "defense", "physical")
            weighted_scores = {}
            for score_key in score_keys:
                values = [(_float(a.get("scores", {}).get(score_key)), weight) for a, weight in confirmed if a.get("scores", {}).get(score_key) is not None]
                if values:
                    weighted_scores[score_key] = round(sum(value * weight for value, weight in values) / sum(weight for _, weight in values), 2)
            base["assessment"] = {
                "scores": weighted_scores,
                "status": "confirmed" if len(weighted_scores) == len(score_keys) else "partial",
                "note": "；".join(dict.fromkeys(str(a.get("note") or "") for a, _ in confirmed if a.get("note"))),
                "source": "human_time_weighted",
            }
        else:
            base["assessment"] = assessments[0] if assessments else {}
        base.pop("_person_key", None)
        merged.append(base)
    for item in merged:
        item.pop("_person_key", None)
    return sorted(merged, key=lambda r: (-r["total_distance_m"], str(r["player_id"])))


def event_timeline(project: dict[str, Any], limit: int = 500) -> list[dict[str, Any]]:
    paths = output_paths(project)
    rows: list[dict[str, Any]] = []
    for r in _read_csv(paths["analysis"] / "pass_events.csv"):
        rows.append({
            "type": "pass", "label": "主动传球候选", "time_sec": _float(r.get("release_time_seconds")), "end_time_sec": _float(r.get("receive_time_seconds")),
            "team_id": r.get("team_id"), "team": _team_label(project, r.get("team_id")), "from_id": _int(r.get("from_global_id"), -1), "to_id": _int(r.get("to_global_id"), -1),
            "distance_m": _float(r.get("distance_m")), "classification": r.get("classification"), "review": "建议复核",
        })
    for r in _read_csv(paths["analysis"] / "possession_transitions.csv"):
        if r.get("classification") == "opponent_possession_change":
            rows.append({
                "type": "turnover", "label": "球权转换", "time_sec": _float(r.get("release_time_seconds")), "end_time_sec": _float(r.get("receive_time_seconds")),
                "team_id": r.get("to_team_id"), "team": _team_label(project, r.get("to_team_id")), "from_id": _int(r.get("from_global_id"), -1), "to_id": _int(r.get("to_global_id"), -1),
                "distance_m": _float(r.get("displacement_m")), "classification": r.get("classification"), "review": "自动事件",
            })
    # Stage-4 events add shots/key actions when available.
    event_candidates = [paths["tracking"] / "tracking" / "events.json", paths["tracking"] / "events.json"]
    for path in event_candidates:
        data = _read_json(path, [])
        if isinstance(data, list):
            for r in data:
                typ = str(r.get("event_type") or r.get("base_event_type") or "关键动作")
                rows.append({"type": "key", "label": typ.replace("_", " / "), "time_sec": _float(r.get("event_time_seconds") or r.get("event_frame_proc")) / (1 if r.get("event_time_seconds") else float(project.get("video", {}).get("fps") or 30)),
                             "end_time_sec": None, "team_id": None, "team": "", "from_id": r.get("primary_global_id"), "to_id": None,
                             "distance_m": None, "classification": typ, "review": r.get("actor_attribution_status") or "候选"})
            break
    semantic_types = {
        "shielding_under_pressure": ("shield", "对抗护球"),
        "counterpress_recovery": ("counterpress", "丢球反抢"),
        "goal_candidate": ("goal", "进球候选"),
    }
    fps = max(1.0, _float((project.get("video") or {}).get("fps"), 30.0))
    for r in semantic_events(project):
        event_type = str(r.get("event_type") or "")
        ui_type, label = semantic_types.get(event_type, ("key", str(r.get("label") or "关键动作")))
        distance = _float((r.get("evidence") or {}).get("minimum_opponent_distance_m"), float("nan"))
        rows.append({
            "type": ui_type, "label": label,
            "time_sec": _int(r.get("event_frame_proc"), -1) / fps,
            "end_time_sec": _float(r.get("end_time")), "team_id": r.get("team_id"),
            "team": _team_label(project, r.get("team_id")),
            "from_id": _int(r.get("primary_global_id"), -1),
            "to_id": _int(r.get("secondary_global_id"), -1),
            "distance_m": distance if math.isfinite(distance) else None,
            "classification": event_type, "review": "候选待复核", "evidence": r.get("evidence") or {},
        })
    rows.sort(key=lambda r: r["time_sec"])
    return rows[: max(1, min(limit, 2000))]


def overview(project: dict[str, Any]) -> dict[str, Any]:
    ps = players(project)
    events = event_timeline(project, 250)
    counts = Counter(e["type"] for e in events)
    return {
        "summary": summary(project), "teams": team_overview(project),
        "leaders": {
            "distance": sorted(ps, key=lambda r: -r["total_distance_m"])[:5],
            "speed": sorted(ps, key=lambda r: -r["max_speed_mps"])[:5],
            "sprints": sorted(ps, key=lambda r: -r["sprint_count"])[:5],
        },
        "event_counts": dict(counts), "timeline": events[:40],
    }


def pitch_data(project: dict[str, Any], max_points_per_player: int = 260) -> dict[str, Any]:
    paths = output_paths(project)
    field_length = float(project.get("settings", {}).get("field_length_m", 45.0)); field_width = float(project.get("settings", {}).get("field_width_m", 25.0))
    calibration_path = project.get("calibration", {}).get("path")
    if calibration_path and Path(calibration_path).is_file():
        cal = _read_json(Path(calibration_path), {}) or {}; b = cal.get("field_bounds_m") or {}
        field_length = _float(b.get("x_max"), field_length) - _float(b.get("x_min"), 0); field_width = _float(b.get("y_max"), field_width) - _float(b.get("y_min"), 0)
    timeseries = _read_csv(paths["running"] / "player_running_timeseries.csv")
    trails: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
    if timeseries:
        for row in timeseries:
            try:
                gid = _int(row.get("global_id"), -1); x = _float(_first_present(row, "x_m_smooth", "x_m", "foot_x_m"), float("nan")); y = _float(_first_present(row, "y_m_smooth", "y_m", "foot_y_m"), float("nan")); frame = _int(_first_present(row, "proc_idx", "frame_proc", "frame"))
                if gid >= 0 and math.isfinite(x) and math.isfinite(y): trails[gid].append((x, y, frame))
            except Exception: continue
    if not trails:
        for row in _read_csv(paths["analysis"] / "possession_frame_evidence.csv"):
            try: trails[_int(row["global_id"])].append((_float(row["player_x_m"]), _float(row["player_y_m"]), _int(row["frame_proc"])))
            except Exception: continue
    team_map = {_int(r.get("global_id"), -1): r.get("team_id", "unassigned") for r in _read_csv(paths["analysis"] / "player_team_map.csv")}
    sampled = []
    for gid, points in sorted(trails.items(), key=lambda kv: -len(kv[1])):
        step = max(1, len(points) // max_points_per_player); arr = points[::step][:max_points_per_player]
        sampled.append({"global_id": gid, "team_id": team_map.get(gid, "unassigned"), "team": _team_label(project, team_map.get(gid)), "points": [[round(x, 3), round(y, 3), f] for x, y, f in arr]})
    pass_lines = []
    for row in _read_csv(paths["analysis"] / "pass_events.csv")[:500]:
        pass_lines.append({"from": _int(row.get("from_global_id"), -1), "to": _int(row.get("to_global_id"), -1), "team_id": row.get("team_id"), "team": _team_label(project, row.get("team_id")),
                           "start": [_float(row.get("start_x_m")), _float(row.get("start_y_m"))], "end": [_float(row.get("end_x_m")), _float(row.get("end_y_m"))], "distance_m": _float(row.get("distance_m")), "time_sec": _float(row.get("release_time_seconds"))})
    return {"field": {"length_m": field_length, "width_m": field_width}, "trails": sampled, "passes": pass_lines,
            "source_note": "完整跑动时序" if timeseries else "持球证据中的真实米制球员位置"}


def _path_version(path: Path) -> tuple[str, int, int]:
    if not path.is_file():
        return "", 0, 0
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=4)
def _replay_frame_index(
    ts_path: str, ts_stamp: int, ts_size: int,
    evidence_path: str, evidence_stamp: int, evidence_size: int,
    team_path: str, team_stamp: int, team_size: int,
    jersey_path: str, jersey_stamp: int, jersey_size: int,
    mot_path: str, mot_stamp: int, mot_size: int,
    ball_path: str, ball_stamp: int, ball_size: int,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, str], dict[int, list[float]], dict[int, list[float]]]:
    """Build the expensive full-match lookup once per output-file version."""
    del ts_stamp, ts_size, evidence_stamp, evidence_size, team_stamp, team_size
    del jersey_stamp, jersey_size, mot_stamp, mot_size, ball_stamp, ball_size
    try:
        team_rows = _read_csv(Path(team_path)) if team_path else []
        team_map = {_int(r.get("global_id"), -1): r.get("team_id", "unassigned") for r in team_rows}
        number_map: dict[int, str] = {}
        for r in (_read_csv(Path(jersey_path)) if jersey_path else []):
            gid = _int(r.get("global_id"), -1)
            status = str(r.get("status") or "").lower()
            num = str(r.get("predicted_number") or "")
            number_map[gid] = num if num and "confirm" in status else ""
        frame_players: dict[int, list[dict[str, Any]]] = defaultdict(list)
        player_lookup: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
        ts = _read_csv(Path(ts_path)) if ts_path else []
        source = ts or (_read_csv(Path(evidence_path)) if evidence_path else [])
        for r in source:
            gid = _int(r.get("global_id"), -1)
            frame = _int(_first_present(r, "proc_idx", "frame_proc"), -1)
            x = _float(_first_present(r, "x_m_smooth", "x_m_raw", "player_x_m"), float("nan"))
            y = _float(_first_present(r, "y_m_smooth", "y_m_raw", "player_y_m"), float("nan"))
            if gid < 0 or frame < 0 or not math.isfinite(x) or not math.isfinite(y):
                continue
            image_x = _float(r.get("foot_x_px"), float("nan"))
            image_y = _float(r.get("foot_y_px"), float("nan"))
            player = {
                "id": gid, "x": round(x, 3), "y": round(y, 3),
                "speed": round(_float(r.get("speed_mps")), 2),
                "team_id": team_map.get(gid, "unassigned"), "number": number_map.get(gid, ""),
            }
            if math.isfinite(image_x) and math.isfinite(image_y):
                player["image"] = [round(image_x, 2), round(image_y, 2)]
            frame_players[frame].append(player)
            player_lookup[frame][gid] = player
        if mot_path:
            with Path(mot_path).open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    parts = line.rstrip().split(",")
                    if len(parts) < 7:
                        continue
                    try:
                        frame = int(float(parts[0])) - 1
                        gid = int(float(parts[1]))
                        player = player_lookup.get(frame, {}).get(gid)
                        if player is not None:
                            player["bbox"] = [
                                round(float(parts[2]), 2), round(float(parts[3]), 2),
                                round(float(parts[4]), 2), round(float(parts[5]), 2),
                                round(float(parts[6]), 3),
                            ]
                    except ValueError:
                        continue
        ball_metric: dict[int, list[float]] = {}
        for r in (_read_csv(Path(evidence_path)) if evidence_path else []):
            frame = _int(r.get("frame_proc"), -1)
            source_name = str(r.get("ball_source") or "observed").lower()
            if frame >= 0 and source_name == "observed" and r.get("ball_x_m") not in (None, ""):
                ball_metric[frame] = [
                    round(_float(r.get("ball_x_m")), 3),
                    round(_float(r.get("ball_y_m")), 3),
                ]
        ball_image: dict[int, list[float]] = {}
        for r in (_read_csv(Path(ball_path)) if ball_path else []):
            frame = _int(r.get("frame_proc"), -1)
            x = _float(r.get("ball_x_px"), float("nan"))
            y = _float(r.get("ball_y_px"), float("nan"))
            if frame >= 0 and math.isfinite(x) and math.isfinite(y):
                ball_image[frame] = [round(x, 2), round(y, 2)]
        return dict(frame_players), team_map, ball_metric, ball_image
    except Exception:
        return {}, {}, {}, {}


def replay_data(
    project: dict[str, Any], max_frames: int | None = None, *,
    start_frame: int | None = None, frame_count: int | None = None,
) -> dict[str, Any]:
    """Browser replay data; windowed requests return every source-video frame."""
    paths = output_paths(project); video = project.get("video") or {}; fps = _float(video.get("fps"), 30.0)
    requested = int(max_frames or project.get("settings", {}).get("replay_max_frames", 15000)); requested = max(120, min(requested, 20000))
    windowed = start_frame is not None
    field_length = _float(project.get("settings", {}).get("field_length_m"), 45.0)
    field_width = _float(project.get("settings", {}).get("field_width_m"), 25.0)
    calibration_path = (project.get("calibration") or {}).get("path")
    if calibration_path and Path(calibration_path).is_file():
        bounds = (_read_json(Path(calibration_path), {}) or {}).get("field_bounds_m") or {}
        field_length = _float(bounds.get("x_max"), field_length) - _float(bounds.get("x_min"), 0.0)
        field_width = _float(bounds.get("y_max"), field_width) - _float(bounds.get("y_min"), 0.0)
    field = {"length_m": field_length, "width_m": field_width}
    index_args = sum((
        _path_version(paths["running"] / "player_running_timeseries.csv"),
        _path_version(paths["analysis"] / "possession_frame_evidence.csv"),
        _path_version(paths["analysis"] / "player_team_map.csv"),
        _path_version(paths["ocr"] / "jersey_number_results.csv"),
        _path_version(paths["tracking"] / "tracking" / "tracking_mot.txt"),
        _path_version(paths["tracking"] / "tracking" / "ball_positions_observed.csv"),
    ), ())
    frame_players, team_map, ball_by_frame, ball_image_by_frame = _replay_frame_index(*index_args)
    if not frame_players:
        return {"field": field, "fps": fps, "duration_seconds": _float(video.get("duration_seconds")), "frame_step": 1, "frames": [], "passes": []}
    minf, maxf = min(frame_players), max(frame_players); span = max(1, maxf - minf + 1)
    total_frames = max(_int(video.get("frame_count"), maxf + 1), maxf + 1)
    step = 1 if windowed else max(1, math.ceil(span / requested))
    range_start = max(0, int(start_frame or minf)) if windowed else minf
    range_end = min(total_frames, range_start + max(60, min(int(frame_count or round(fps * 60)), 3600))) if windowed else maxf + 1
    # Stable possession is expanded only across confirmed intervals. Transfer
    # gaps intentionally have no owner, so the possession halo disappears.
    owner_by_frame: dict[int, int] = {}
    possession_intervals = _read_csv(paths["analysis"] / "possession_intervals.csv")
    if not possession_intervals:
        # Legacy result sets may only have frame evidence. Preserve their owner
        # marker while new runs use stable intervals and explicit transfer gaps.
        for r in _read_csv(paths["analysis"] / "possession_frame_evidence.csv"):
            frame = _int(r.get("frame_proc"), -1)
            gid = _int(r.get("global_id"), -1)
            if range_start <= frame < range_end and gid >= 0:
                owner_by_frame[frame] = gid
    for r in possession_intervals:
        gid = _int(r.get("global_id"), -1)
        start = max(range_start, _int(r.get("start_frame_proc"), -1)); end = min(range_end - 1, _int(r.get("end_frame_proc"), -1))
        if gid >= 0 and start >= 0 and end >= start:
            for frame in range(start, end + 1): owner_by_frame[frame] = gid
    frames = []
    available = sorted(frame_players)
    owner_keys = sorted(owner_by_frame)
    def nearest(keys: list[int], target: int) -> int | None:
        if not keys: return None
        pos = bisect_left(keys, target)
        choices = keys[max(0, pos - 1):min(len(keys), pos + 1)]
        return min(choices, key=lambda value: abs(value - target)) if choices else None
    for f in range(range_start, range_end, step):
        if windowed:
            chosen = f
        else:
            # exact frame preferred, otherwise nearest observed frame within the sampling window
            position = bisect_left(available, f)
            if position >= len(available) or available[position] >= min(f + step, maxf + 1): continue
            chosen = available[position]
        # A football marker is evidence, not a prediction. Never carry the last
        # observed ball across a frame where the detector did not see it.
        ball = ball_by_frame.get(chosen)
        ball_image = ball_image_by_frame.get(chosen)
        owner = owner_by_frame.get(chosen)
        if not windowed:
            nearby_owner = nearest(owner_keys, chosen)
            if owner is None and nearby_owner is not None and abs(nearby_owner - chosen) <= max(step, int(fps)):
                owner = owner_by_frame[nearby_owner]
        frames.append({"frame": chosen, "time_sec": round(chosen / fps, 4), "players": frame_players.get(chosen, []), "ball": ball, "ball_image": ball_image, "possession_id": owner})
    passes = [{"time_sec": _float(r.get("release_time_seconds")), "end_time_sec": _float(r.get("receive_time_seconds")), "from": _int(r.get("from_global_id"), -1), "to": _int(r.get("to_global_id"), -1), "team_id": r.get("team_id"),
               "start": [_float(r.get("start_x_m")), _float(r.get("start_y_m"))], "end": [_float(r.get("end_x_m")), _float(r.get("end_y_m"))]} for r in _read_csv(paths["analysis"] / "pass_events.csv")]
    ball_observations = [[round(frame / fps, 4), point[0], point[1]] for frame, point in sorted(ball_image_by_frame.items()) if not windowed or range_start <= frame < range_end]
    return {"field": field, "fps": fps, "duration_seconds": _float(video.get("duration_seconds"), maxf / fps), "frame_step": step, "frames": frames, "passes": passes, "ball_observations": ball_observations,
            "total_frames": total_frames, "window_start": range_start, "window_end": range_end,
            "sampling_mode": "source_frame" if windowed else "sampled",
            "team_labels": {tid: _team_label(project, tid) for tid in sorted(set(team_map.values()))}}


def heatmap_points(project: dict[str, Any], global_ids: list[int]) -> list[list[float]]:
    paths = output_paths(project); ids = set(global_ids); points = []
    ts = _read_csv(paths["running"] / "player_running_timeseries.csv")
    for row in ts:
        gid = _int(row.get("global_id"), -1)
        if gid not in ids: continue
        x = _float(_first_present(row, "x_m_smooth", "x_m_raw"), float("nan")); y = _float(_first_present(row, "y_m_smooth", "y_m_raw"), float("nan"))
        if math.isfinite(x) and math.isfinite(y): points.append([x, y])
    if not points:
        for row in _read_csv(paths["analysis"] / "possession_frame_evidence.csv"):
            if _int(row.get("global_id"), -1) in ids: points.append([_float(row.get("player_x_m")), _float(row.get("player_y_m"))])
    step = max(1, len(points) // 2000)
    return [[round(x, 3), round(y, 3)] for x, y in points[::step][:2000]]


def heatmap_image_path(project: dict[str, Any], global_ids: list[int]) -> Path | None:
    """Use formal-card heatmaps when compact result retention removed raw time series.

    Linked technical IDs are blended by their effective playing time, matching
    the weighted aggregation used by the player centre's numeric metrics.
    """
    paths = output_paths(project)
    sources: list[tuple[Path, float]] = []
    for gid in sorted({int(value) for value in global_ids}):
        folder = paths["formal_cards"] / f"unknown_{gid}"
        image = folder / "heatmap.png"
        if not image.is_file():
            continue
        running = _read_json(folder / "running.json", {}) or {}
        summary_row = running.get("summary") if isinstance(running, dict) else {}
        weight = max(1.0, _float((summary_row or {}).get("playing_time_sec"), 1.0))
        sources.append((image, weight))
    if not sources:
        return None
    if len(sources) == 1:
        return sources[0][0]

    from PIL import Image

    cache_dir = paths["root"] / ".browser_cache" / "heatmaps"
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = "_".join(str(int(gid)) for gid in sorted({int(value) for value in global_ids}))
    destination = cache_dir / f"linked_{signature}.png"
    newest_source = max(path.stat().st_mtime_ns for path, _ in sources)
    if destination.is_file() and destination.stat().st_mtime_ns >= newest_source:
        return destination

    merged = Image.open(sources[0][0]).convert("RGB")
    total_weight = sources[0][1]
    for source, weight in sources[1:]:
        layer = Image.open(source).convert("RGB")
        if layer.size != merged.size:
            layer = layer.resize(merged.size, Image.Resampling.LANCZOS)
        merged = Image.blend(merged, layer, weight / (total_weight + weight))
        total_weight += weight
    temporary = destination.with_name(f".{destination.name}.tmp.png")
    merged.save(temporary, format="PNG", optimize=True)
    temporary.replace(destination)
    return destination


def player_events(project: dict[str, Any], global_ids: list[int], limit: int = 1000) -> list[dict[str, Any]]:
    """Return every event involving any technical ID linked to one player."""
    ids = {int(gid) for gid in global_ids}
    return [
        row for row in event_timeline(project, limit)
        if _int(row.get("from_id"), -1) in ids or _int(row.get("to_id"), -1) in ids
    ]


def player_visibility_intervals(project: dict[str, Any], global_ids: list[int]) -> list[tuple[float, float]]:
    """Build merged source-time intervals in which a linked ID is visible."""
    paths = output_paths(project); ids = {int(gid) for gid in global_ids}
    fps = max(1.0, _float((project.get("video") or {}).get("fps"), 30.0))
    frames = sorted({
        _int(row.get("proc_idx"), -1) for row in _read_csv(paths["running"] / "player_running_timeseries.csv")
        if _int(row.get("global_id"), -1) in ids and _int(row.get("proc_idx"), -1) >= 0
    })
    if not frames:
        return []
    # Bridge short tracking gaps so the compilation remains watchable while
    # preserving every visible period from every linked technical ID.
    max_gap = max(1, int(round(fps * 1.5)))
    ranges: list[list[int]] = [[frames[0], frames[0]]]
    for frame in frames[1:]:
        if frame - ranges[-1][1] <= max_gap:
            ranges[-1][1] = frame
        else:
            ranges.append([frame, frame])
    return [(max(0.0, start / fps - 0.35), (end + 1) / fps + 0.35) for start, end in ranges]


def player_compilation_manifest(project: dict[str, Any], global_ids: list[int]) -> dict[str, Any]:
    """Map source-time MOT boxes into the concatenated player-review timeline."""
    gids = {int(gid) for gid in global_ids}
    fps = max(1.0, _float((project.get("video") or {}).get("fps"), 30.0))
    intervals = player_visibility_intervals(project, list(gids))
    timeline = []
    cursor = 0.0
    for start, end in intervals:
        timeline.append({"source_start": start, "source_end": end, "compilation_start": cursor, "duration": end - start})
        cursor += end - start
    boxes = []
    mot_path = output_paths(project)["tracking"] / "tracking" / "tracking_mot.txt"
    sample_stride = max(1, int(round(fps / 15.0)))
    if mot_path.is_file() and timeline:
        with mot_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                parts = line.rstrip().split(",")
                if len(parts) < 7:
                    continue
                try:
                    frame = int(float(parts[0])) - 1; gid = int(float(parts[1]))
                except ValueError:
                    continue
                if gid not in gids or frame % sample_stride:
                    continue
                source_time = frame / fps
                segment = next((row for row in timeline if row["source_start"] <= source_time < row["source_end"]), None)
                if segment is None:
                    continue
                boxes.append({
                    "time_sec": round(segment["compilation_start"] + source_time - segment["source_start"], 3),
                    "source_time_sec": round(source_time, 3), "global_id": gid,
                    "bbox": [round(float(parts[2]), 2), round(float(parts[3]), 2), round(float(parts[4]), 2), round(float(parts[5]), 2)],
                })
    return {"global_ids": sorted(gids), "duration_seconds": round(cursor, 3), "fps": fps, "intervals": timeline, "boxes": boxes}


def highlights(project: dict[str, Any]) -> list[dict[str, Any]]:
    paths = output_paths(project); items: list[Path] = []
    if paths["highlights"].is_dir(): items.extend(sorted(paths["highlights"].glob("*.mp4")))
    if project.get("kind") == "demo" and not items and paths.get("demo_video", Path("")).is_file(): items.append(paths["demo_video"])
    # Backward-compatible search for older formal runs.
    if project.get("kind") != "demo" and not items:
        for root in [paths["tracking"] / "id_focus", paths["tracking"] / "candidate_highlights", paths["cards"]]:
            if root.exists(): items.extend(sorted(root.rglob("*.mp4")))
    manifest = _read_json(paths["highlights"] / "id_focus_clips.json", []) or []
    by_file = {r.get("clip_file"): r for r in manifest if isinstance(r, dict)}
    seen = set(); out=[]
    for path in items:
        key = str(path.resolve())
        if key in seen: continue
        seen.add(key); meta = by_file.get(path.name, {}); gid = meta.get("global_id") or meta.get("primary_global_id")
        out.append({"name": f"球员高光 · TARGET ID {gid}" if gid is not None else path.stem, "path": key, "target_labeled": gid is not None or "target_player_" in path.name.lower() or "gid_" in path.name.lower(),
                    "global_id": gid, "event_id": meta.get("event_id"), "duration_seconds": meta.get("duration_seconds"), "base_event_type": meta.get("base_event_type")})
        if len(out) >= 24: break
    return out


def quality_summary(project: dict[str, Any]) -> dict[str, Any]:
    paths = output_paths(project); cal = project.get("calibration") or {}
    running_q = _read_json(paths["running"] / "running_quality_report.json", {}) or {}
    analysis_q = _read_json(paths["analysis"] / "quality_report.json", {}) or {}
    tracking_q = _read_json(paths["tracking"] / "quality_report.json", {}) or _read_json(paths["tracking"] / "tracking" / "quality_report.json", {}) or {}
    ocr = _read_csv(paths["ocr"] / "jersey_number_results.csv"); ocr_counts = Counter(str(r.get("status") or "unknown") for r in ocr)
    pass_review = _read_csv(paths["analysis"] / "acceptance_sample_20.csv")
    labeled = sum(1 for r in pass_review if str(r.get("human_is_pass") or "").strip())
    identity_audit = _read_json(paths.get("identity_audit", Path()) / "audit_report.json", {}) or {}
    identity_audit_error = _read_json(paths.get("identity_audit", Path()) / "audit_error.json", {}) or {}
    identity_state = None
    assessment_state = None
    if project.get("kind") != "demo":
        try:
            from app.services.reviews import load_pass_review, load_identity_review, load_player_assessments
            review_state = load_pass_review(project)
            identity_state = load_identity_review(project)
            assessment_state = load_player_assessments(project)
            pass_review = review_state.get("rows", [])
            labeled = int(review_state.get("labeled", 0))
        except Exception:
            review_state = None
            identity_state = None
            assessment_state = None
    else:
        review_state = None
    return {
        "calibration": {"status": cal.get("status"), "message": cal.get("message"), "validation": cal.get("validation"), "anchors": cal.get("anchors") or []},
        "tracking": {"available": bool(tracking_q), "report": tracking_q},
        "running": {"available": bool(running_q), "calibration_validation_passed": running_q.get("calibration_validation_passed"), "identities": len(running_q.get("identities") or [])},
        "analysis": {"available": bool(analysis_q), "status": analysis_q.get("status") or "available", "report": analysis_q},
        "jersey": {"total": len(ocr), "status_counts": dict(ocr_counts)},
        "pass_review": {"sample_size": len(pass_review), "labeled": labeled, "status": (review_state.get("status") if review_state else ("complete" if pass_review and labeled == len(pass_review) else "pending")), "agreement_rate": (review_state.get("agreement_rate") if review_state else None)},
        "identity_review": ({"confirmed": identity_state.get("confirmed", 0), "total": identity_state.get("total", 0), "status": identity_state.get("status", "pending")} if identity_state else {"confirmed": 0, "total": 0, "status": "read_only" if project.get("kind") == "demo" else "pending"}),
        "player_assessment": ({"confirmed": assessment_state.get("confirmed", 0), "total": assessment_state.get("total", 0), "status": assessment_state.get("status", "pending")} if assessment_state else {"confirmed": 0, "total": 0, "status": "read_only" if project.get("kind") == "demo" else "pending"}),
        "identity_audit": {
            "available": bool(identity_audit),
            "status": identity_audit.get("status") or identity_audit_error.get("status") or "not_run",
            "ids_with_transitions": int((identity_audit.get("counts") or {}).get("ids_with_transitions") or 0),
            "detected_transitions": int((identity_audit.get("counts") or {}).get("detected_transitions") or 0),
            "message": identity_audit_error.get("message") or "身份审计只产生复核候选，不自动修改技术 ID。",
        },
        "product_note": "质量页用于判断哪些结果可以直接展示、哪些仍建议人工复核；不会把候选结果包装成已确认事实。",
    }


def report_preview_path(project: dict[str, Any]) -> Path | None:
    paths = output_paths(project); path = paths["report_html"]
    return path if path.is_file() else None


def replay_video_path(project: dict[str, Any]) -> Path | None:
    path = output_paths(project)["replay_video"]
    return path if path.is_file() else None


def _build_curated_exports(project: dict[str, Any], paths: dict[str, Path]) -> list[tuple[str, str, Path]]:
    """Materialize exports that reflect human ID linking, not raw ID counts."""
    if project.get("kind") == "demo":
        return []
    root = paths["root"] / "curated_exports"; root.mkdir(parents=True, exist_ok=True)
    roster_path = root / "players_merged.csv"
    event_path = root / "player_events_merged.json"
    player_rows = players(project)
    fields = ["player_name", "team", "jersey_number", "technical_global_ids", "technical_id_count", "identity_status", "total_distance_m", "visible_time_sec", "sprint_count", "max_speed_mps", "assessment_status"]
    with roster_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for item in player_rows:
            writer.writerow({
                "player_name": item.get("player_id"), "team": item.get("team"), "jersey_number": item.get("jersey_number"),
                "technical_global_ids": ";".join(map(str, item.get("global_ids") or [])), "technical_id_count": len(item.get("global_ids") or []),
                "identity_status": item.get("identity_status"), "total_distance_m": item.get("total_distance_m"),
                "visible_time_sec": item.get("visible_time_sec"), "sprint_count": item.get("sprint_count"), "max_speed_mps": item.get("max_speed_mps"),
                "assessment_status": (item.get("assessment") or {}).get("status", "pending"),
            })
    event_payload = [{
        "player_name": item.get("player_id"), "global_ids": item.get("global_ids") or [],
        "events": player_events(project, item.get("global_ids") or []),
    } for item in player_rows]
    event_path.write_text(json.dumps(event_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return [("已关联球员汇总 CSV", "人工复核", roster_path), ("按球员聚合事件 JSON", "事件数据", event_path)]


def downloadable_files(project: dict[str, Any]) -> list[dict[str, Any]]:
    paths = output_paths(project); out=[]
    candidates = [
        ("比赛分析报告", "报告", paths["report_html"]),
        ("2D 赛后回放", "视频", paths["replay_video"]),
        ("传球事件 CSV", "事件数据", paths["analysis"] / "pass_events.csv"),
        ("传球矩阵 CSV", "事件数据", paths["analysis"] / "pass_matrix_long.csv"),
        ("球权区间 CSV", "事件数据", paths["analysis"] / "possession_intervals.csv"),
        ("球员跑动汇总 CSV", "球员数据", paths["running"] / "player_running_summary.csv"),
        ("球员跑动时序 CSV", "球员数据", paths["running"] / "player_running_timeseries.csv"),
        ("号码识别 CSV", "球员数据", paths["ocr"] / "jersey_number_results.csv"),
        ("分析质量报告 JSON", "质检", paths["analysis"] / "quality_report.json"),
        ("跑动质量报告 JSON", "质检", paths["running"] / "running_quality_report.json"),
        ("身份质量审计 JSON", "质检", paths.get("identity_audit", Path()) / "audit_report.json"),
        ("身份切换候选 CSV", "质检", paths.get("identity_audit", Path()) / "transitions.csv"),
        ("动态标定 JSON", "标定", Path(project.get("calibration", {}).get("path") or "")),
        ("结果完整性清单 JSON", "归档", paths.get("artifact_manifest", Path(""))),
    ]
    if project.get("kind") != "demo":
        review_root = project_dir(project["id"]) / "reviews"
        candidates.extend([
            ("传球人工复核 JSON", "人工复核", review_root / "pass_review.json"),
            ("球员身份映射 JSON", "人工复核", review_root / "identity_mapping.json"),
            ("球员八维人工评估 JSON", "人工复核", review_root / "player_assessments.json"),
            ("球员正式报告标注 JSON", "人工复核", review_root / "player_report_annotations.json"),
        ])
        candidates.extend(_build_curated_exports(project, paths))
        for path in sorted((paths["root"] / "player_reports").glob("*.pdf")) if (paths["root"] / "player_reports").is_dir() else []:
            candidates.append((f"单球员正式报告 · {path.stem}", "报告", path))
        for path in sorted((paths["root"] / "player_compilations").glob("*.mp4")) if (paths["root"] / "player_compilations").is_dir() else []:
            candidates.append((f"球员总视频 · {path.stem}", "视频", path))
    if project.get("kind") == "demo": candidates.append(("球员评估 PDF", "报告", paths["report_pdf"]))
    for label, category, path in candidates:
        if path and path.is_file(): out.append({"label": label, "category": category, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "filename": path.name})
    return out
