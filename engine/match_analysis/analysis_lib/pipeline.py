from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

from .acceptance import evaluate_annotations, make_sample_rows, sha256
from .geometry import HomographyProvider, _portable_basename, validate_calibration_compatibility
from .io import read_ball, read_metadata, read_mot, read_team_map, write_csv
from .passes import (build_statistics, detect_active_passes, detect_pass_review_candidates,
                     detect_possession_transitions)
from .possession import match_ball_to_players, stable_possessions
from .teams import assign_teams_kmeans, diagnostics_from_explicit_map


@dataclass(frozen=True)
class PipelineConfig:
    possession_distance_m: float = 1.5
    possession_min_frames: int = 3
    max_ball_gap_frames: int = 2
    max_transfer_gap_seconds: float = 1.5
    min_pass_displacement_m: float = .5
    pass_review_min_displacement_m: float = .25
    team_clusters: int = 2
    team_samples_per_id: int = 12
    acceptance_sample_size: int = 20
    acceptance_threshold: float = .80

    def validate(self) -> None:
        if self.possession_distance_m <= 0:
            raise ValueError("possession_distance_m 必须 > 0")
        if self.possession_min_frames < 3:
            raise ValueError("本周口径要求 possession_min_frames >= 3")
        if self.max_ball_gap_frames < 0 or self.max_transfer_gap_seconds < 0:
            raise ValueError("gap 参数不能为负")
        if self.min_pass_displacement_m <= 0:
            raise ValueError("min_pass_displacement_m 必须 > 0")
        if not 0 < self.pass_review_min_displacement_m < self.min_pass_displacement_m:
            raise ValueError("pass_review_min_displacement_m 必须 > 0 且小于正式传球阈值")
        if self.team_clusters not in (2, 3):
            raise ValueError("team_clusters 仅支持 2 或 3")
        if self.acceptance_sample_size != 20:
            raise ValueError("本周验收固定抽测 20 条")
        if not 0 <= self.acceptance_threshold <= 1:
            raise ValueError("acceptance_threshold 必须在 0..1")


def run_analysis(
    *, tracking_dir: str | Path, calibration: str | Path, output: str | Path,
    video: str | Path | None = None, team_map_path: str | Path | None = None,
    annotations: str | Path | None = None, config: PipelineConfig | None = None,
    fps_override: float | None = None, vid_stride_override: int | None = None,
) -> dict:
    config = config or PipelineConfig()
    config.validate()
    tracking_dir = Path(tracking_dir).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"输出目录必须不存在，避免覆盖既有结果: {output}")
    mot = _resolve(tracking_dir, ["tracking_mot.txt", "tracking/tracking_mot.txt"])
    ball = _resolve(tracking_dir, ["ball_positions_observed.csv", "tracking/ball_positions_observed.csv"])
    metadata_path = _resolve_optional(tracking_dir, ["tracking_run_metadata.json", "tracking/tracking_run_metadata.json"])
    metadata = read_metadata(metadata_path)
    fps = float(fps_override or metadata.get("processed_fps", 0))
    if fps <= 0:
        raise ValueError("缺少 processed_fps；请提供带元数据的追踪目录或 --fps")
    vid_stride = int(vid_stride_override or metadata.get("vid_stride", 1))
    if vid_stride < 1:
        raise ValueError("vid_stride 必须 >= 1")
    provider = HomographyProvider(calibration)
    video_name = _portable_basename(metadata.get("video")) or (Path(video).name if video else None)
    width = height = None
    if video is not None:
        import cv2
        cap = cv2.VideoCapture(str(video)); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None; cap.release()
    validate_calibration_compatibility(
        provider, expected_video_name=video_name, width=width, height=height, vid_stride=vid_stride,
    )
    players_by_frame, mot_rows = read_mot(mot)
    ball_by_frame = read_ball(ball, config.max_ball_gap_frames)
    identities = {row.global_id for row in mot_rows}
    if team_map_path:
        team_map = read_team_map(team_map_path)
        team_diagnostics = diagnostics_from_explicit_map(team_map, identities)
    else:
        if video is None:
            raise ValueError("未提供 --team-map 时必须提供 --video 运行轨迹级球衣 K-means")
        team_map, team_diagnostics = assign_teams_kmeans(
            video, mot_rows, config.team_clusters, config.team_samples_per_id,
        )
    missing = identities - team_map.keys()
    for identity in missing:
        team_map[identity] = "unassigned"

    work = output.with_name(output.name + f".work-{int(time.time())}")
    if work.exists():
        raise FileExistsError(work)
    work.mkdir(parents=True)
    started = time.time()
    try:
        matches, projection_report = match_ball_to_players(
            players_by_frame, ball_by_frame, provider, team_map,
            max_distance_m=config.possession_distance_m, vid_stride=vid_stride,
        )
        intervals, stable_frames = stable_possessions(matches, config.possession_min_frames)
        transitions = detect_possession_transitions(
            intervals, round(config.max_transfer_gap_seconds * fps),
            short_displacement_m=config.min_pass_displacement_m,
        )
        events = detect_active_passes(
            transitions, min_displacement_m=config.min_pass_displacement_m, fps=fps,
        )
        review_candidates = detect_pass_review_candidates(
            transitions,
            review_min_displacement_m=config.pass_review_min_displacement_m,
            formal_min_displacement_m=config.min_pass_displacement_m,
        )
        team_summary, matrix_long, matrix_json = build_statistics(events)
        transition_rows = []
        for transition in transitions:
            row = asdict(transition)
            row["release_time_seconds"] = round(transition.release_frame_proc / fps, 3)
            row["receive_time_seconds"] = round(transition.receive_frame_proc / fps, 3)
            row["displacement_m"] = round(transition.displacement_m, 3)
            transition_rows.append(row)
        event_rows = []
        for event in events:
            row = asdict(event)
            row["release_time_seconds"] = round(event.release_frame_proc / fps, 3)
            row["receive_time_seconds"] = round(event.receive_frame_proc / fps, 3)
            row["distance_m"] = round(event.distance_m, 3)
            event_rows.append(row)
        write_csv(work / "player_team_map.csv", team_diagnostics, [
            "global_id", "team_id", "samples", "nearest_center_distance", "center_margin", "assignment_method",
        ])
        write_csv(work / "possession_frame_evidence.csv", [{**asdict(row), "distance_m": round(row.distance_m, 4)} for row in stable_frames], [
            "frame_proc", "global_id", "team_id", "distance_m", "ball_x_m", "ball_y_m",
            "player_x_m", "player_y_m", "ball_source",
        ])
        write_csv(work / "possession_intervals.csv", [asdict(row) for row in intervals], list(asdict(intervals[0]).keys()) if intervals else [
            "possession_id", "global_id", "team_id", "start_frame_proc", "confirmed_frame_proc",
            "end_frame_proc", "evidence_frames", "min_distance_m", "max_distance_m", "mean_distance_m",
            "start_ball_x_m", "start_ball_y_m", "end_ball_x_m", "end_ball_y_m",
        ])
        transition_columns = list(transition_rows[0].keys()) if transition_rows else [
            "transition_id", "from_global_id", "to_global_id", "from_team_id", "to_team_id",
            "release_frame_proc", "receive_frame_proc", "receive_confirmed_frame_proc",
            "transfer_gap_frames", "source_evidence_frames", "receiver_evidence_frames",
            "start_x_m", "start_y_m", "end_x_m", "end_y_m", "dx_m", "dy_m",
            "displacement_m", "classification", "release_time_seconds", "receive_time_seconds",
        ]
        write_csv(work / "possession_transitions.csv", transition_rows, transition_columns)
        pass_columns = list(event_rows[0].keys()) if event_rows else [
            "pass_id", "transition_id", "from_global_id", "to_global_id", "team_id",
            "release_frame_proc", "receive_frame_proc", "receive_confirmed_frame_proc", "transfer_gap_frames",
            "start_x_m", "start_y_m", "end_x_m", "end_y_m", "dx_m", "dy_m", "distance_m",
            "direction_angle_deg", "transfer_speed_mps", "classification", "intent_proxy",
            "release_time_seconds", "receive_time_seconds",
        ]
        write_csv(work / "pass_events.csv", event_rows, pass_columns)
        review_rows = []
        for candidate in review_candidates:
            row = dict(candidate)
            row["release_time_seconds"] = round(row["release_frame_proc"] / fps, 3)
            row["receive_time_seconds"] = round(row["receive_frame_proc"] / fps, 3)
            row["displacement_m"] = round(row["displacement_m"], 3)
            review_rows.append(row)
        review_columns = list(review_rows[0].keys()) if review_rows else [
            "transition_id", "from_global_id", "to_global_id", "from_team_id", "to_team_id",
            "release_frame_proc", "receive_frame_proc", "receive_confirmed_frame_proc",
            "transfer_gap_frames", "source_evidence_frames", "receiver_evidence_frames",
            "start_x_m", "start_y_m", "end_x_m", "end_y_m", "dx_m", "dy_m",
            "displacement_m", "classification", "review_classification", "review_reason",
            "release_time_seconds", "receive_time_seconds",
        ]
        write_csv(work / "pass_review_candidates.csv", review_rows, review_columns)
        write_csv(work / "team_pass_summary.csv", team_summary, [
            "team_id", "active_directed_passes", "total_pass_distance_m", "mean_pass_distance_m",
        ])
        write_csv(work / "pass_matrix_long.csv", matrix_long, [
            "team_id", "from_global_id", "to_global_id", "active_directed_passes",
        ])
        (work / "pass_matrices.json").write_text(json.dumps(matrix_json, ensure_ascii=False, indent=2), encoding="utf-8")
        sample_rows = make_sample_rows(events, config.acceptance_sample_size, fps)
        write_csv(work / "acceptance_sample_20.csv", sample_rows, [
            "pass_id", "event_time_seconds", "from_global_id", "to_global_id", "from_team_id", "to_team_id",
            "model_outcome", "distance_m", "human_is_pass", "human_outcome", "human_note",
        ])
        human = evaluate_annotations(
            sample_rows, annotations, config.acceptance_threshold, config.acceptance_sample_size,
        )
        geometric_valid = all(
            event.receive_confirmed_frame_proc - event.receive_frame_proc + 1 >= config.possession_min_frames
            and event.transfer_gap_frames >= 0 for event in events
        )
        report = {
            "schema_version": 1, "status": human["status"],
            "task_contract": {
                "possession": f"ball-to-player-footpoint distance < {config.possession_distance_m} m for >= {config.possession_min_frames} consecutive frames",
                "possession_transition": "every stable A-to-B possession change within the transfer gap, including opponents and short/ID-jitter candidates",
                "pass_network": f"same-team stable A-to-B transfer with metric displacement >= {config.min_pass_displacement_m} m; tactical-intent proxy, human review required",
                "acceptance": f"deterministic time-spread sample of 20; human agreement >= {config.acceptance_threshold:.0%}",
            },
            "inputs": {
                "tracking_dir": str(tracking_dir), "mot": str(mot), "mot_sha256": sha256(mot),
                "ball": str(ball), "ball_sha256": sha256(ball), "calibration": str(provider.info.source),
                "calibration_sha256": sha256(provider.info.source), "video": str(Path(video).resolve()) if video else None,
                "team_map": str(Path(team_map_path).resolve()) if team_map_path else None,
            },
            "frame_domain": {"processed_fps": fps, "vid_stride": vid_stride, "mot_is_1_based": True, "internal_frame_proc_is_0_based": True},
            "counts": {
                "mot_rows": len(mot_rows), "global_ids": len(identities), "ball_points_after_bounded_interpolation": len(ball_by_frame),
                "stable_possession_frames": len(stable_frames), "possession_intervals": len(intervals),
                "possession_transitions": len(transitions), "pass_events": len(events),
                "pass_review_gray_zone": len(review_candidates),
                "opponent_possession_changes": sum(t.classification == "opponent_possession_change" for t in transitions),
                "short_or_stationary_same_team_changes": sum(t.classification == "same_team_short_or_stationary_transition" for t in transitions),
                "unknown_team_transitions": sum(t.classification == "unknown_team_possession_change" for t in transitions),
            },
            "projection": projection_report,
            "geometry_checks": {"all_passes_have_confirmed_receiver_and_nonnegative_gap": geometric_valid},
            "human_acceptance": human,
            "warnings": [
                "global_id remains a run-local candidate identity; ID switches directly affect the pass matrix.",
                "K-means team numbers are cluster labels, not semantic home/away names.",
                "Possession transitions and pass-network edges are separate outputs; turnovers and ID jitter never enter the pass matrix.",
                "The 0.25-0.5m same-team gray zone is exported only for human review and never counted in the formal pass network.",
                "Tactical intent is represented only by an auditable same-team+stable-control+metric-displacement proxy, not claimed as ground truth.",
            ],
            "elapsed_seconds": round(time.time() - started, 3),
        }
        (work / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (work / "quality_report.md").write_text(_markdown_report(report), encoding="utf-8")
        manifest_files = sorted(path for path in work.iterdir() if path.is_file())
        manifest = {
            "pipeline": "match_analysis_passing_network_alpha", "status": report["status"],
            "config": asdict(config),
            "artifacts": {path.name: sha256(path) for path in manifest_files},
        }
        (work / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        work.rename(output)
        return report
    except Exception:
        # Preserve the staging directory for diagnosis; never delete partial evidence automatically.
        raise


def _resolve(root: Path, candidates: list[str]) -> Path:
    result = _resolve_optional(root, candidates)
    if result is None:
        raise FileNotFoundError(f"在 {root} 中找不到任一输入: {candidates}")
    return result


def _resolve_optional(root: Path, candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path
    return None


def _markdown_report(report: dict) -> str:
    human = report["human_acceptance"]
    rate = "待人工填写" if human["agreement_rate"] is None else f"{human['agreement_rate']:.1%}"
    return f"""# match analysis 传球网络质检报告

- 状态：`{report['status']}`
- 球权片段：{report['counts']['possession_intervals']}
- 全部球权转换：{report['counts']['possession_transitions']}
- 主动定向传球候选：{report['counts']['pass_events']}
- 仅供复核的灰区候选：{report['counts']['pass_review_gray_zone']}
- 异队球权转换：{report['counts']['opponent_possession_changes']}
- 同队短距离/静止切换：{report['counts']['short_or_stationary_same_team_changes']}
- 人工有效标注：{human['valid_labels']}/{human['required_labels']}
- 人机一致率：{rate}（门槛 {human['threshold']:.0%}）

## 固定口径

{report['task_contract']['possession']}。所有 A→B 稳定变化进入球权转换表；只有满足 `{report['task_contract']['pass_network']}` 的事件进入传球表和传球网络。抽样只从传球候选中按时间确定性覆盖，不使用球权转换凑足样本。

## 解释边界

global_id 仍是单次追踪候选身份；队伍编号是 K-means 簇标签。战术意图无法由单目轨迹直接证明，当前只输出可审计候选。只有候选数量达到 20、填写 `acceptance_sample_20.csv` 并重新带入验收后，报告才可能通过。
"""
