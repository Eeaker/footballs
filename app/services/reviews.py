from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services.storage import project_dir


@lru_cache(maxsize=24)
def _csv_snapshot(path_text: str, modified_ns: int, size_bytes: int) -> tuple[dict[str, str], ...]:
    del modified_ns, size_bytes
    with Path(path_text).open("r", encoding="utf-8-sig", newline="") as f:
        return tuple(csv.DictReader(f))


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    stat = path.stat()
    return list(_csv_snapshot(str(path.resolve()), stat.st_mtime_ns, stat.st_size))


def _outputs(project: dict[str, Any]) -> Path:
    # Reviews are writable only for formal projects. Demo support is read-only and
    # intentionally returns no review rows.
    return project_dir(project["id"]) / "outputs"


def _pass_source(project: dict[str, Any]) -> list[dict[str, str]]:
    root = _outputs(project) / "match_analysis" / "analysis"
    source = _csv(root / "acceptance_sample_20.csv")
    return source[:20] if source else _csv(root / "pass_events.csv")[:20]


def _pass_store_path(project: dict[str, Any]) -> Path:
    return project_dir(project["id"]) / "reviews" / "pass_review.json"


def load_pass_review(project: dict[str, Any]) -> dict[str, Any]:
    if project.get("kind") == "demo":
        return {"rows": [], "labeled": 0, "total": 0, "agreement_rate": None, "status": "read_only"}
    source = _pass_source(project)
    stored: dict[str, Any] = {}
    path = _pass_store_path(project)
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8")).get("labels", {})
        except Exception:
            stored = {}
    rows = []
    for idx, row in enumerate(source[:20]):
        key = str(row.get("pass_id") or row.get("event_id") or idx)
        label = stored.get(key, {})
        try:
            time_sec = float(row.get("release_time_seconds") or row.get("event_time_seconds") or 0)
        except Exception:
            time_sec = 0.0
        try:
            distance_m = float(row.get("distance_m") or row.get("displacement_m") or 0)
        except Exception:
            distance_m = 0.0
        rows.append({
            "index": idx,
            "key": key,
            "pass_id": row.get("pass_id"),
            "time_sec": time_sec,
            "from_global_id": row.get("from_global_id"),
            "to_global_id": row.get("to_global_id"),
            "team_id": row.get("team_id"),
            "distance_m": distance_m,
            "machine_is_pass": True,
            "human_is_pass": label.get("human_is_pass"),
            "outcome": label.get("outcome", ""),
            "note": label.get("note", ""),
        })
    labeled = sum(r["human_is_pass"] is not None for r in rows)
    agreement = sum(
        bool(r["human_is_pass"]) == bool(r["machine_is_pass"])
        for r in rows if r["human_is_pass"] is not None
    )
    return {
        "rows": rows,
        "labeled": labeled,
        "total": len(rows),
        "agreement_rate": agreement / labeled if labeled else None,
        "status": "complete" if rows and labeled == len(rows) else "pending",
    }


def save_pass_review_label(
    project: dict[str, Any], key: str, human_is_pass: bool | None,
    outcome: str = "", note: str = ""
) -> dict[str, Any]:
    if project.get("kind") == "demo":
        raise ValueError("示例项目为只读")
    valid_keys = {str(r["key"]) for r in load_pass_review(project)["rows"]}
    if str(key) not in valid_keys:
        raise KeyError(key)
    path = _pass_store_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema_version": 1, "labels": {}}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    labels = payload.setdefault("labels", {})
    if human_is_pass is None and not outcome and not note:
        labels.pop(str(key), None)
    else:
        labels[str(key)] = {
            "human_is_pass": human_is_pass,
            "outcome": outcome[:40],
            "note": note[:500],
        }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_pass_review(project)


def _identity_store_path(project: dict[str, Any]) -> Path:
    return project_dir(project["id"]) / "reviews" / "identity_mapping.json"


def _mapping_person_key(mapping: dict[str, Any]) -> str:
    key = str(mapping.get("person_key") or "").strip()
    if key:
        return key
    if mapping.get("roster_index") is not None:
        return f"roster:{mapping['roster_index']}"
    if str(mapping.get("name") or "").strip():
        return "manual:" + "|".join(str(mapping.get(name) or "").strip().casefold() for name in ("team_id", "jersey_number", "name"))
    return ""


def _load_roster(project: dict[str, Any]) -> list[dict[str, Any]]:
    roster = project.get("roster") or {}
    path = Path(str(roster.get("path") or ""))
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            src = data if isinstance(data, list) else data.get("players", [])
            rows = [dict(x) for x in src if isinstance(x, dict)]
        else:
            rows = _csv(path)
    except Exception:
        return []
    normalized = []
    for i, row in enumerate(rows):
        # Accept common Chinese/English headers without silently inventing data.
        name = row.get("name") or row.get("player_name") or row.get("姓名") or row.get("球员") or ""
        number = row.get("jersey_number") or row.get("number") or row.get("号码") or row.get("球衣号码") or ""
        team = row.get("team") or row.get("team_name") or row.get("球队") or ""
        normalized.append({"index": i, "name": str(name), "jersey_number": str(number), "team": str(team), "raw": row})
    return normalized


def _candidate_global_ids(project: dict[str, Any]) -> list[int]:
    root = _outputs(project)
    analysis = root / "match_analysis" / "analysis"
    running = root / "match_analysis" / "metric_running"
    ids: set[int] = set()
    for row in _csv(analysis / "player_team_map.csv") + _csv(running / "player_running_summary.csv"):
        try:
            ids.add(int(float(row.get("global_id", ""))))
        except Exception:
            continue
    return sorted(ids)


def load_identity_review(project: dict[str, Any]) -> dict[str, Any]:
    if project.get("kind") == "demo":
        return {"mappings": {}, "roster": [], "candidate_global_ids": [], "confirmed": 0, "total": 0, "status": "read_only"}
    path = _identity_store_path(project)
    mappings: dict[str, Any] = {}
    if path.is_file():
        try:
            mappings = json.loads(path.read_text(encoding="utf-8")).get("mappings", {})
        except Exception:
            mappings = {}
    ids = _candidate_global_ids(project)
    confirmed = sum(1 for gid in ids if str(gid) in mappings and mappings[str(gid)].get("name"))
    return {
        "mappings": mappings,
        "roster": _load_roster(project),
        "candidate_global_ids": ids,
        "confirmed": confirmed,
        "total": len(ids),
        "status": "complete" if ids and confirmed == len(ids) else "partial" if confirmed else "pending",
    }


def _identity_merge_evidence(project: dict[str, Any]) -> tuple[dict[int, str], dict[int, set[int]]]:
    """Load only the evidence needed to prevent impossible identity merges."""
    root = _outputs(project)
    teams: dict[int, str] = {}
    for row in _csv(root / "match_analysis" / "analysis" / "player_team_map.csv"):
        try:
            teams[int(float(row.get("global_id", "")))] = str(row.get("team_id") or "").strip()
        except Exception:
            continue
    frames: dict[int, set[int]] = {}
    for row in _csv(root / "match_analysis" / "metric_running" / "player_running_timeseries.csv"):
        try:
            gid = int(float(row.get("global_id", "")))
            frame = int(float(row.get("proc_idx", "")))
        except Exception:
            continue
        frames.setdefault(gid, set()).add(frame)
    return teams, frames


def identity_merge_candidates(project: dict[str, Any], global_id: int) -> dict[str, Any]:
    """Explain which technical IDs may be linked to the selected identity."""
    valid_ids = _candidate_global_ids(project)
    if global_id not in valid_ids:
        raise KeyError(global_id)
    teams, frames = _identity_merge_evidence(project)
    mappings = load_identity_review(project).get("mappings", {})
    current = mappings.get(str(global_id), {})
    current_key = _mapping_person_key(current)
    current_number = str(current.get("jersey_number") or "").strip()
    target_frames = frames.get(global_id, set())
    rows = []
    for candidate in valid_ids:
        reasons: list[str] = []
        if candidate != global_id:
            left_team, right_team = teams.get(global_id, ""), teams.get(candidate, "")
            if left_team and right_team and left_team != right_team:
                reasons.append(f"不同球队：{left_team} / {right_team}")
            overlap = len(target_frames & frames.get(candidate, set()))
            if overlap:
                reasons.append(f"同一时间同时出现：{overlap} 帧")
            other = mappings.get(str(candidate), {})
            other_key = _mapping_person_key(other)
            if other_key and current_key and other_key != current_key:
                reasons.append("已关联到另一名已确认球员")
            other_number = str(other.get("jersey_number") or "").strip()
            if current_number and other_number and current_number != other_number:
                reasons.append(f"已确认号码冲突：{current_number} / {other_number}")
        rows.append({
            "global_id": candidate,
            "team_id": teams.get(candidate, ""),
            "compatible": not reasons,
            "reasons": reasons,
            "currently_linked": bool(current_key and _mapping_person_key(mappings.get(str(candidate), {})) == current_key),
        })
    return {"global_id": global_id, "rules": ["不同球队不可合并", "同一时间同时出现的技术 ID 不可合并", "已确认到其他球员或号码冲突时不可合并"], "candidates": rows}


def save_identity_mapping(
    project: dict[str, Any], global_id: int, *, name: str = "", jersey_number: str = "",
    team_id: str = "", roster_index: int | None = None, note: str = "",
    linked_global_ids: list[int] | None = None,
) -> dict[str, Any]:
    if project.get("kind") == "demo":
        raise ValueError("示例项目为只读")
    valid_ids = set(_candidate_global_ids(project))
    if global_id not in valid_ids:
        raise KeyError(global_id)
    path = _identity_store_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema_version": 1, "mappings": {}}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    mappings = payload.setdefault("mappings", {})
    targets = sorted(set(linked_global_ids or [global_id]))
    if global_id not in targets:
        targets.append(global_id)
        targets.sort()
    if not set(targets).issubset(valid_ids):
        raise KeyError(next(gid for gid in targets if gid not in valid_ids))
    if len(targets) > 1:
        compatibility = {row["global_id"]: row for row in identity_merge_candidates(project, global_id)["candidates"]}
        rejected = [compatibility[gid] for gid in targets if gid != global_id and not compatibility.get(gid, {}).get("compatible")]
        if rejected:
            detail = "；".join(f"ID {row['global_id']}：{'、'.join(row['reasons'])}" for row in rejected)
            raise ValueError(f"不能合并这些技术 ID。{detail}")
    if not any([name.strip(), jersey_number.strip(), team_id.strip(), note.strip(), roster_index is not None]):
        mappings.pop(str(global_id), None)
    else:
        if roster_index is not None:
            person_key = f"roster:{roster_index}"
        elif name.strip():
            person_key = "manual:" + "|".join(x.strip().casefold() for x in (team_id, jersey_number, name))
        else:
            person_key = "linked:" + ",".join(str(gid) for gid in targets)
        previous_key = _mapping_person_key(mappings.get(str(global_id)) or {})
        if previous_key:
            for mapped_gid in list(mappings):
                if mapped_gid not in {str(gid) for gid in targets} and _mapping_person_key(mappings.get(mapped_gid) or {}) == previous_key:
                    mappings.pop(mapped_gid, None)
        entry = {
            "name": name.strip()[:80],
            "jersey_number": jersey_number.strip()[:20],
            "team_id": team_id.strip()[:40],
            "roster_index": roster_index,
            "note": note.strip()[:500],
            "person_key": person_key[:240],
            "linked_global_ids": targets,
        }
        for gid in targets:
            mappings[str(gid)] = dict(entry)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_identity_review(project)


REPORT_ANNOTATION_FIELDS = (
    "position", "preferred_foot", "club", "nickname", "quote",
    "strengths_summary", "improvements_summary", "style_tag", "style_narrative",
    "reference_player", "potential_grade", "potential_direction",
    "tactical_literacy", "physical_competition", "talent",
    "similarities", "differences", "next_target",
    "position_1", "position_1_fit", "position_1_description", "position_1_verdict",
    "position_2", "position_2_fit", "position_2_description", "position_2_verdict",
    "position_3", "position_3_fit", "position_3_description", "position_3_verdict",
    "to_player", "to_family_and_coach",
)


def _report_annotation_path(project: dict[str, Any]) -> Path:
    return project_dir(project["id"]) / "reviews" / "player_report_annotations.json"


def _report_person_key(project: dict[str, Any], global_id: int) -> str:
    mapping = load_identity_review(project).get("mappings", {}).get(str(global_id), {})
    return _mapping_person_key(mapping) or f"technical:{global_id}"


def load_player_report_annotation(project: dict[str, Any], global_id: int) -> dict[str, Any]:
    if global_id not in set(_candidate_global_ids(project)):
        raise KeyError(global_id)
    path = _report_annotation_path(project)
    payload = {"schema_version": 1, "annotations": {}}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    key = _report_person_key(project, global_id)
    values = payload.get("annotations", {}).get(key, {})
    return {"person_key": key, "global_id": global_id, "fields": {name: str(values.get(name) or "") for name in REPORT_ANNOTATION_FIELDS}}


def save_player_report_annotation(project: dict[str, Any], global_id: int, fields: dict[str, Any]) -> dict[str, Any]:
    if project.get("kind") == "demo":
        raise ValueError("示例项目为只读")
    current = load_player_report_annotation(project, global_id)
    clean = {name: str(fields.get(name) or "").strip()[:2000] for name in REPORT_ANNOTATION_FIELDS}
    path = _report_annotation_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema_version": 1, "annotations": {}}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    payload.setdefault("annotations", {})[current["person_key"]] = clean
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_player_report_annotation(project, global_id)


def identity_mapping_dict(project: dict[str, Any]) -> dict[str, Any]:
    if project.get("kind") == "demo":
        return {}
    return load_identity_review(project).get("mappings", {})


ASSESSMENT_DIMENSIONS = (
    "speed", "endurance", "running", "passing", "control", "shooting", "defense", "physical",
)


def _assessment_store_path(project: dict[str, Any]) -> Path:
    return project_dir(project["id"]) / "reviews" / "player_assessments.json"


def load_player_assessments(project: dict[str, Any]) -> dict[str, Any]:
    """Load human-validated eight-dimension player assessments.

    Scores are deliberately human-entered 0-100 values. The system never fills
    missing dimensions with inferred/fake values. A player is ``confirmed`` only
    after all eight dimensions are present.
    """
    if project.get("kind") == "demo":
        return {"assessments": {}, "dimensions": list(ASSESSMENT_DIMENSIONS), "confirmed": 0, "total": 0, "status": "read_only"}
    path = _assessment_store_path(project)
    assessments: dict[str, Any] = {}
    if path.is_file():
        try:
            assessments = json.loads(path.read_text(encoding="utf-8")).get("assessments", {})
        except Exception:
            assessments = {}
    ids = _candidate_global_ids(project)
    confirmed = 0
    normalized: dict[str, Any] = {}
    for gid in ids:
        raw = assessments.get(str(gid), {}) if isinstance(assessments, dict) else {}
        scores = raw.get("scores", {}) if isinstance(raw, dict) else {}
        clean_scores: dict[str, float] = {}
        for key in ASSESSMENT_DIMENSIONS:
            if key not in scores or scores.get(key) in (None, ""):
                continue
            try:
                value = float(scores[key])
            except Exception:
                continue
            if 0 <= value <= 100:
                clean_scores[key] = round(value, 1)
        status = "confirmed" if len(clean_scores) == len(ASSESSMENT_DIMENSIONS) else "partial" if clean_scores else "pending"
        if status == "confirmed":
            confirmed += 1
        if raw or clean_scores:
            normalized[str(gid)] = {
                "scores": clean_scores,
                "status": status,
                "note": str(raw.get("note") or "")[:500] if isinstance(raw, dict) else "",
                "source": "human",
            }
    return {
        "assessments": normalized,
        "dimensions": list(ASSESSMENT_DIMENSIONS),
        "confirmed": confirmed,
        "total": len(ids),
        "status": "complete" if ids and confirmed == len(ids) else "partial" if confirmed or normalized else "pending",
    }


def save_player_assessment(
    project: dict[str, Any], global_id: int, *, scores: dict[str, Any] | None = None, note: str = ""
) -> dict[str, Any]:
    if project.get("kind") == "demo":
        raise ValueError("示例项目为只读")
    if global_id not in set(_candidate_global_ids(project)):
        raise KeyError(global_id)
    scores = scores or {}
    unknown = set(scores) - set(ASSESSMENT_DIMENSIONS)
    if unknown:
        raise ValueError("存在未知评分维度：" + ", ".join(sorted(unknown)))
    clean: dict[str, float] = {}
    for key in ASSESSMENT_DIMENSIONS:
        value = scores.get(key)
        if value in (None, ""):
            continue
        try:
            num = float(value)
        except Exception as exc:
            raise ValueError(f"{key} 必须是 0–100 的数字") from exc
        if not 0 <= num <= 100:
            raise ValueError(f"{key} 必须在 0–100 之间")
        clean[key] = round(num, 1)

    path = _assessment_store_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema_version": 1, "dimensions": list(ASSESSMENT_DIMENSIONS), "assessments": {}}
    if path.is_file():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(old, dict):
                payload.update(old)
                payload.setdefault("assessments", {})
        except Exception:
            pass
    entries = payload.setdefault("assessments", {})
    if not clean and not note.strip():
        entries.pop(str(global_id), None)
    else:
        entries[str(global_id)] = {
            "scores": clean,
            "note": note.strip()[:500],
            "source": "human",
        }
    payload["schema_version"] = 1
    payload["dimensions"] = list(ASSESSMENT_DIMENSIONS)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_player_assessments(project)


def player_assessment_dict(project: dict[str, Any]) -> dict[str, Any]:
    if project.get("kind") == "demo":
        return {}
    return load_player_assessments(project).get("assessments", {})
