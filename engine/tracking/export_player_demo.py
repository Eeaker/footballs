from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import subprocess

import cv2


TEAM_ENGLISH = {"白": "WHITE", "黄": "YELLOW", "蓝": "BLUE", "红": "RED"}
EVENT_ENGLISH = {
    "射门_大力踢球": "SHOT / POWER KICK",
    "传球_解围_方向突变": "PASS / CLEARANCE",
    "关键动作": "KEY ACTION",
}


def parse_player(text: str) -> tuple[str, str]:
    """解析“队伍:号码”形式的球员选择。"""
    parts = text.replace("：", ":").split(":", 1)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise ValueError(f"球员应写成 队伍:号码，例如 白:20，收到 {text!r}")
    return parts[0].strip(), parts[1].strip()


def load_identity_map(path: Path) -> dict[int, dict]:
    """读取临时身份表；保留置信度和排除项，不擅自补全未知号码。"""
    result: dict[int, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            gid = int(row["global_id"])
            team = (row.get("队伍") or "").strip()
            number = (row.get("号码") or "").strip()
            confidence = (row.get("置信度") or "").strip()
            excluded = team == "排除" or confidence == "排除"
            result[gid] = {
                "global_id": gid,
                "team": None if excluded or not team else team,
                "number": None if excluded or not number else number,
                "canonical_key": None if excluded or not team or not number else f"{team}_{number}",
                "excluded": excluded,
                "confidence": confidence,
                "merge_note": (row.get("归并说明") or "").strip(),
                "note": (row.get("备注") or "").strip(),
            }
    if not result:
        raise ValueError(f"身份表为空: {path}")
    return result


def read_events(path: Path, fps: float) -> list[dict]:
    """读取 V3 的 CSV 或 JSON 事件索引并统一字段类型。"""
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    result = []
    for raw in rows:
        row = dict(raw)
        row["event_id"] = int(row["event_id"])
        row["event_frame_proc"] = int(float(row["event_frame_proc"]))
        row["score"] = float(row.get("score") or 0.0)
        row["primary_global_id"] = (
            None if row.get("primary_global_id") in (None, "", "None")
            else int(float(row["primary_global_id"]))
        )
        row["event_time_seconds"] = row["event_frame_proc"] / fps
        result.append(row)
    return result


def read_mot(path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """读取 MOT，并将帧号转换为从零开始的处理帧。"""
    frames: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.rstrip().split(",")
            if len(fields) < 6:
                raise ValueError(f"{path}:{line_number} MOT字段不足")
            frames[int(float(fields[0])) - 1].append((
                int(float(fields[1])), float(fields[2]), float(fields[3]),
                float(fields[4]), float(fields[5]),
            ))
    return dict(frames)


def select_player_events(events: list[dict], candidate_ids: set[int], count: int = 5,
                         min_gap_seconds: float = 8.0) -> list[dict]:
    """优先选择主体自动归属且类型分散的五个事件。"""
    rows = [
        row for row in events
        if row["primary_global_id"] in candidate_ids
        and row.get("actor_attribution_status") == "auto"
    ]
    if len(rows) < count:
        rows = [row for row in events if row["primary_global_id"] in candidate_ids]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("base_event_type") or "关键动作")].append(row)
    for values in by_type.values():
        values.sort(key=lambda row: (-row["score"], row["event_id"]))
    type_order = ["射门_大力踢球", "传球_解围_方向突变", "关键动作"]
    type_order += sorted(set(by_type) - set(type_order))
    selected: list[dict] = []

    def allowed(row: dict) -> bool:
        return all(abs(row["event_time_seconds"] - old["event_time_seconds"]) >= min_gap_seconds
                   for old in selected)

    while len(selected) < count:
        added = False
        for event_type in type_order:
            candidate = next((row for row in by_type.get(event_type, [])
                              if row not in selected and allowed(row)), None)
            if candidate is not None:
                selected.append(candidate)
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
    if len(selected) < count:
        for row in sorted(rows, key=lambda item: (-item["score"], item["event_id"])):
            if row not in selected:
                selected.append(row)
            if len(selected) == count:
                break
    return selected


def alias_overlap_report(mot: dict, candidate_ids: set[int], fps: float) -> dict:
    """检查被人工归并的候选ID是否在同一帧同时出现。"""
    overlaps = []
    maximum = 0
    for frame, boxes in mot.items():
        present = {gid for gid, *_ in boxes if gid in candidate_ids}
        if len(present) > 1:
            overlaps.append(int(frame))
            maximum = max(maximum, len(present))
    return {
        "candidate_global_ids": sorted(candidate_ids),
        "overlap_frames": len(overlaps),
        "overlap_seconds": round(len(overlaps) / max(fps, 1e-9), 3),
        "max_simultaneous_candidate_ids": maximum,
        "conflict": bool(overlaps),
        "render_policy": "highlight_event_source_global_id_only",
    }


def audit_identity_map(identity: dict[int, dict], mot: dict, fps: float) -> dict:
    """审计身份表中的全部多人归并，防止把同帧共现者静默视为同一人。"""
    groups: dict[str, set[int]] = defaultdict(set)
    for gid, row in identity.items():
        if row["canonical_key"]:
            groups[row["canonical_key"]].add(gid)
    rows = []
    for player_key, candidate_ids in sorted(groups.items()):
        if len(candidate_ids) < 2:
            continue
        row = {"player_key": player_key, **alias_overlap_report(mot, candidate_ids, fps)}
        rows.append(row)
    return {
        "multi_candidate_player_groups": len(rows),
        "conflicting_groups": sum(row["conflict"] for row in rows),
        "groups": rows,
        "warning": "同帧共现表示候选ID不能作为逐帧轨迹直接无条件归并；Demo按事件原始主体ID高亮。",
    }


def clip_bounds(event_seconds: float, duration_seconds: float, before: float,
                after: float) -> tuple[float, float]:
    """计算不越过视频边界的事件窗口。"""
    return max(0.0, event_seconds - before), min(duration_seconds, event_seconds + after)


def _draw_overlay(frame, boxes, candidate_ids: set[int], label: str, event: dict,
                  event_order: int, total_events: int) -> None:
    height, width = frame.shape[:2]
    target_seen = False
    for gid, x, y, w, h in boxes:
        if gid not in candidate_ids:
            continue
        target_seen = True
        p1 = (max(0, int(round(x))), max(0, int(round(y))))
        p2 = (min(width - 1, int(round(x + w))), min(height - 1, int(round(y + h))))
        cv2.rectangle(frame, p1, p2, (0, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (p1[0], max(30, p1[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, .82, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (width, 74), (20, 20, 20), -1)
    event_type = EVENT_ENGLISH.get(str(event.get("base_event_type")), "KEY ACTION")
    title = f"{label}   EVENT {event_order}/{total_events}   {event_type}"
    cv2.putText(frame, title, (22, 32), cv2.FONT_HERSHEY_SIMPLEX, .72,
                (255, 255, 255), 2, cv2.LINE_AA)
    status = "TARGET TRACKED" if target_seen else "TARGET TEMPORARILY NOT VISIBLE"
    cv2.putText(frame, status, (22, 62), cv2.FONT_HERSHEY_SIMPLEX, .56,
                (70, 240, 70) if target_seen else (70, 180, 255), 2, cv2.LINE_AA)


def _mux_audio(source_video: Path, silent_video: Path, output: Path,
               segments: list[tuple[float, float]]) -> bool:
    """把原视频对应时间段的音频拼接回标注视频；无 ffmpeg 时安全降级。"""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            ffmpeg = None
    if ffmpeg is None:
        silent_video.replace(output)
        return False
    filters = []
    labels = []
    for index, (start, end) in enumerate(segments):
        filters.append(
            f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[a{index}]"
        )
        labels.append(f"[a{index}]")
    filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[aout]")
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(source_video), "-i", str(silent_video),
        "-filter_complex", ";".join(filters), "-map", "1:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", str(output),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        silent_video.replace(output)
        return False
    silent_video.unlink()
    return True


def render_player(video: Path, mot: dict, events: list[dict], candidate_ids: set[int],
                  team: str, number: str, output_dir: Path, source_before: float = 15.0,
                  source_after: float = 15.0, edit_before: float = 4.0,
                  edit_after: float = 2.0, vid_stride: int = 1) -> dict:
    """生成五段30秒候选、150秒候选串联和默认30秒精编成品。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fps = raw_fps / max(1, vid_stride)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = raw_frames / raw_fps
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / "candidates_5x30"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    label = f"{TEAM_ENGLISH.get(team, team)} #{number}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    source_windows = [clip_bounds(row["event_time_seconds"], duration, source_before, source_after)
                      for row in events]
    edit_windows = [clip_bounds(row["event_time_seconds"], duration, edit_before, edit_after)
                    for row in events]
    source_silent = []
    source_writers = []
    for order, row in enumerate(events, 1):
        path = candidate_dir / f".event_{order:02d}_{row['event_id']:03d}_silent.mp4"
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"无法创建候选视频: {path}")
        source_silent.append(path); source_writers.append(writer)
    compilation_silent = output_dir / ".candidate_compilation_150s_silent.mp4"
    edit_silent = output_dir / ".player_demo_30s_silent.mp4"
    compilation_writer = cv2.VideoWriter(str(compilation_silent), fourcc, fps, (width, height))
    edit_writer = cv2.VideoWriter(str(edit_silent), fourcc, fps, (width, height))
    if not compilation_writer.isOpened() or not edit_writer.isOpened():
        raise RuntimeError("无法创建球员集锦视频")

    raw_index = -1
    proc_index = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_index += 1
        if (raw_index + 1) % max(1, vid_stride) != 0:
            continue
        proc_index += 1
        seconds = raw_index / raw_fps
        boxes = mot.get(proc_index, [])
        for index, row in enumerate(events):
            source_start, source_end = source_windows[index]
            if source_start <= seconds < source_end:
                canvas = frame.copy()
                _draw_overlay(canvas, boxes, {int(row["primary_global_id"])},
                              label, row, index + 1, len(events))
                source_writers[index].write(canvas)
                compilation_writer.write(canvas)
            edit_start, edit_end = edit_windows[index]
            if edit_start <= seconds < edit_end:
                canvas = frame.copy()
                _draw_overlay(canvas, boxes, {int(row["primary_global_id"])},
                              label, row, index + 1, len(events))
                edit_writer.write(canvas)
    cap.release()
    for writer in source_writers:
        writer.release()
    compilation_writer.release(); edit_writer.release()

    candidate_rows = []
    audio_muxed = True
    for index, (row, silent, window) in enumerate(zip(events, source_silent, source_windows), 1):
        final = candidate_dir / f"event_{index:02d}_{row['event_id']:03d}_30s.mp4"
        audio_muxed &= _mux_audio(video, silent, final, [window])
        candidate_rows.append({
            "order": index, "event_id": row["event_id"],
            "candidate_global_id": row["primary_global_id"],
            "event_type": row.get("base_event_type"), "score": row["score"],
            "event_time_seconds": round(row["event_time_seconds"], 3),
            "source_start_seconds": round(window[0], 3),
            "source_end_seconds": round(window[1], 3), "file": str(final.relative_to(output_dir)),
        })
    candidate_compilation = output_dir / "candidate_compilation_5x30_150s.mp4"
    final_edit = output_dir / "player_demo_final_30s.mp4"
    audio_muxed &= _mux_audio(video, compilation_silent, candidate_compilation, source_windows)
    audio_muxed &= _mux_audio(video, edit_silent, final_edit, edit_windows)
    manifest = {
        "player": {"team": team, "number": number, "label": label,
                   "candidate_global_ids": sorted(candidate_ids)},
        "identity_alias_audit": alias_overlap_report(mot, candidate_ids, fps),
        "policy": {
            "candidate_events": len(events), "seconds_before_each_event": source_before,
            "seconds_after_each_event": source_after,
            "candidate_compilation_seconds": round(sum(end - start for start, end in source_windows), 3),
            "default_final_edit_seconds": round(sum(end - start for start, end in edit_windows), 3),
            "final_edit_per_event": {"before_seconds": edit_before, "after_seconds": edit_after},
            "multimodal_direct_scoring": False,
        },
        "events": candidate_rows,
        "candidate_compilation_file": candidate_compilation.name,
        "final_edit_file": final_edit.name,
        "audio_muxed": audio_muxed,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def export_mapped_events(events: list[dict], identity: dict[int, dict], output: Path) -> dict:
    """导出应用临时 identity map 后的完整事件表。"""
    rows = []
    for event in events:
        row = dict(event)
        mapping = identity.get(event["primary_global_id"]) if event["primary_global_id"] is not None else None
        row["candidate_global_id"] = event["primary_global_id"]
        row["team"] = mapping["team"] if mapping else None
        row["number"] = mapping["number"] if mapping else None
        row["player_key"] = mapping["canonical_key"] if mapping else None
        row["identity_confidence"] = mapping["confidence"] if mapping else None
        row["identity_review_required"] = not bool(mapping and mapping["canonical_key"]
                                                   and mapping["confidence"] in {"确定", "较确定"})
        rows.append(row)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "events": len(rows),
        "mapped_to_numbered_player": sum(row["player_key"] is not None for row in rows),
        "identity_review_required": sum(row["identity_review_required"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="基于tracking V3和临时identity map导出两名球员Demo")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--identity-map", type=Path, required=True)
    parser.add_argument("--player", action="append", required=True, help="可重复，例如 --player 白:20")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--source-before", type=float, default=15.0)
    parser.add_argument("--source-after", type=float, default=15.0)
    parser.add_argument("--edit-before", type=float, default=4.0)
    parser.add_argument("--edit-after", type=float, default=2.0)
    parser.add_argument("--vid-stride", type=int, default=1)
    args = parser.parse_args()
    if args.count < 1 or min(args.source_before, args.source_after, args.edit_before, args.edit_after) < 0:
        parser.error("事件数量必须>=1，窗口秒数必须>=0")
    if abs(args.count * (args.edit_before + args.edit_after) - 30.0) > .01:
        parser.error("默认精编总时长必须为30秒")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")
    raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0); cap.release()
    fps = raw_fps / max(1, args.vid_stride)
    identity = load_identity_map(args.identity_map)
    events = read_events(args.events, fps)
    mot = read_mot(args.mot)
    args.outdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.identity_map, args.outdir / "identity_map_source.csv")
    map_report = export_mapped_events(events, identity, args.outdir / "events_identity_mapped.json")
    consistency = audit_identity_map(identity, mot, fps)
    (args.outdir / "identity_map_consistency_report.json").write_text(
        json.dumps(consistency, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifests = []
    for raw_player in args.player:
        team, number = parse_player(raw_player)
        candidate_ids = {
            gid for gid, row in identity.items()
            if row["team"] == team and row["number"] == number and not row["excluded"]
        }
        if not candidate_ids:
            raise ValueError(f"identity map中找不到球员 {team}:{number}")
        selected = select_player_events(events, candidate_ids, args.count)
        if len(selected) < args.count:
            raise ValueError(f"球员 {team}:{number} 只有 {len(selected)} 个可用事件，不足 {args.count}")
        # 成品按比赛时间顺序播放；事件强度只负责入选，不能让单次顺序和画面标题错位。
        selected.sort(key=lambda row: (row["event_time_seconds"], row["event_id"]))
        directory = args.outdir / f"player_{TEAM_ENGLISH.get(team, team).lower()}_{number}"
        manifests.append(render_player(
            args.video, mot, selected, candidate_ids, team, number, directory,
            args.source_before, args.source_after, args.edit_before, args.edit_after,
            args.vid_stride,
        ))
    report = {
        "status": "complete", "source_video": str(args.video.resolve()),
        "source_mot": str(args.mot.resolve()), "source_events": str(args.events.resolve()),
        "identity_map": map_report, "identity_map_consistency": consistency,
        "players": manifests,
    }
    (args.outdir / "demo_delivery_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "players": len(manifests),
                      "output": str(args.outdir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
