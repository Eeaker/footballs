import json

from app.services.pipeline import (
    _clear_downstream, _highlights_complete, _internal_player_cards_complete, _player_card_progress, _player_cards_complete,
    _video_output_complete,
    _select_focus_events, _tracking_progress,
)


def test_tracking_detection_progress_is_bounded_before_post_processing():
    progress, message = _tracking_progress("    处理到帧 31000, 当前 tracklet 数 42", 62000)
    assert progress == 41
    assert "31,000/62,000" in message


def test_tracking_render_progress_uses_final_tracking_band():
    progress, message = _tracking_progress("    渲染到帧 31000, 总帧 62000", 62000)
    assert progress == 88
    assert "追踪回放" in message


def test_tracking_stage_milestones_are_reported():
    assert _tracking_progress("[Field] tracklet 9 -> 8", 10)[0] == 79
    assert _tracking_progress("[Stage 2] 全局重关联", 10)[0] == 81
    assert _tracking_progress("[Stage 3] 渲染可视化视频", 10)[0] == 83
    assert _tracking_progress("[Stage 4] 事件检测与切片", 10)[0] == 94


def test_unrelated_output_does_not_rewrite_progress():
    assert _tracking_progress("loading model", 100) is None


def test_player_card_delivery_reports_per_player_progress():
    progress, message = _player_card_progress("FT player cards: players=10/40")
    assert progress == 13
    assert "10/40 人" in message
    assert _player_card_progress("ffmpeg output") is None


def test_focus_events_accept_machine_event_envelope_and_time_fields(tmp_path):
    source = tmp_path / "events.json"
    destination = tmp_path / "focus.json"
    source.write_text(json.dumps({
        "schema_version": "machine_events_v1",
        "events": [{
            "event_id": "pass_0042",
            "start_time": 10.0,
            "end_time": 13.5,
            "anchor_time": 11.25,
            "primary_global_id": 7,
            "event_type": "active_directed_pass_candidate",
            "confidence": 0.8,
        }],
    }), encoding="utf-8")

    assert _select_focus_events(source, destination, limit=3, fps=30.0) == 1
    selected = json.loads(destination.read_text(encoding="utf-8"))
    assert selected[0]["source_event_id"] == "pass_0042"
    assert selected[0]["event_id"] == 0
    assert selected[0]["start_frame_proc"] == 300
    assert selected[0]["event_frame_proc"] == 338
    assert selected[0]["end_frame_proc"] == 405


def test_focus_events_prioritize_requested_semantic_candidates(tmp_path):
    source = tmp_path / "events.json"
    destination = tmp_path / "focus.json"
    source.write_text(json.dumps({"events": [
        {"event_id": "ordinary", "start_time": 1, "end_time": 2, "primary_global_id": 1, "event_type": "active_directed_pass_candidate", "confidence": .99},
        {"event_id": "shield", "start_time": 3, "end_time": 4, "primary_global_id": 1, "event_type": "shielding_under_pressure"},
        {"event_id": "press", "start_time": 5, "end_time": 6, "primary_global_id": 1, "event_type": "counterpress_recovery"},
        {"event_id": "goal", "start_time": 7, "end_time": 8, "primary_global_id": 1, "event_type": "goal_candidate"},
    ]}, ensure_ascii=False), encoding="utf-8")

    assert _select_focus_events(source, destination, limit=3, fps=30) == 3
    selected = json.loads(destination.read_text(encoding="utf-8"))
    assert [row["source_event_id"] for row in selected] == ["goal", "press", "shield"]


def test_report_retry_preserves_completed_player_cards(tmp_path):
    outputs = tmp_path / "outputs"
    (outputs / "player_cards").mkdir(parents=True)
    (outputs / "player_cards" / "package_manifest.json").write_text("{}", encoding="utf-8")
    (outputs / "player_cards" / "summary.txt").write_text("done", encoding="utf-8")
    (outputs / "player_cards_formal").mkdir()
    (outputs / "player_cards_formal" / "summary.txt").write_text("done", encoding="utf-8")
    (outputs / "highlights").mkdir()

    assert _player_cards_complete(outputs)
    _clear_downstream(outputs, "report", preserve_player_cards=True)

    assert (outputs / "player_cards" / "package_manifest.json").is_file()
    assert (outputs / "player_cards_formal" / "summary.txt").is_file()
    assert not (outputs / "highlights").exists()


def test_report_retry_can_preserve_partial_formal_export(tmp_path):
    outputs = tmp_path / "outputs"
    (outputs / "player_cards").mkdir(parents=True)
    (outputs / "player_cards" / "package_manifest.json").write_text("{}", encoding="utf-8")
    (outputs / "player_cards" / "summary.txt").write_text("done", encoding="utf-8")
    partial = outputs / "player_cards_formal" / "unknown_25"
    partial.mkdir(parents=True)
    (partial / "events_for_annotation.json").write_text("{}", encoding="utf-8")
    (outputs / "highlights").mkdir()

    assert _internal_player_cards_complete(outputs)
    assert not _player_cards_complete(outputs)
    _clear_downstream(outputs, "report", preserve_player_cards=True)

    assert (partial / "events_for_annotation.json").is_file()
    assert not (outputs / "highlights").exists()


def test_completed_report_media_can_be_reused_without_deleting_open_files(tmp_path):
    outputs = tmp_path / "outputs"
    highlights = outputs / "highlights"
    highlights.mkdir(parents=True)
    clip = highlights / "event_0004_gid_002.mp4"
    clip.write_bytes(b"v" * 2048)
    (highlights / "id_focus_clips.json").write_text(json.dumps([
        {"clip_file": clip.name, "event_id": 4, "global_id": 2}
    ]), encoding="utf-8")
    replay = outputs / "metric_pitch_replay.mp4"
    replay.write_bytes(b"r" * 2048)

    assert _highlights_complete(outputs)
    assert _video_output_complete(replay)
    _clear_downstream(
        outputs,
        "report",
        preserve_highlights=True,
        preserve_metric_replay=True,
    )

    assert clip.is_file()
    assert replay.is_file()
