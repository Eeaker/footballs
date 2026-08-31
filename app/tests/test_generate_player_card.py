from __future__ import annotations

import argparse
import sys
from pathlib import Path


MATCH_ANALYSIS_ROOT = Path(__file__).resolve().parents[2] / "engine" / "match_analysis"
if str(MATCH_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(MATCH_ANALYSIS_ROOT))

import generate_player_card  # noqa: E402
import export_player_card_delivery_v1 as formal_export  # noqa: E402


def test_precomputed_running_can_keep_mot_provenance(monkeypatch, tmp_path: Path) -> None:
    """The formal exporter needs MOT even when running metrics are precomputed."""
    mot = tmp_path / "tracking_mot.txt"
    captured: dict = {}
    args = argparse.Namespace(
        video=tmp_path / "source.mp4",
        mot=mot,
        numbers=tmp_path / "clip_eligibility.json",
        run_number_ocr=False,
        team_hints=None,
        number_ocr_output=None,
        ocr_cpu=False,
        ocr_maximum_candidates_per_id=36,
        ocr_reuse_candidates=None,
        events=tmp_path / "events.json",
        calibration=tmp_path / "calibration.json",
        running_timeseries=tmp_path / "running.csv",
        fps=30.0,
        running_src=tmp_path / "running_src",
        output=tmp_path / "cards",
        formal_output=None,
    )
    monkeypatch.setattr(generate_player_card, "parse_args", lambda: args)
    monkeypatch.setattr(
        generate_player_card,
        "generate_from_mot",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must reuse precomputed running")),
    )

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return {"players": [], "event_count": 0}

    monkeypatch.setattr(generate_player_card, "generate_player_card_data", fake_generate)
    generate_player_card.main()

    assert captured["running_timeseries"] == args.running_timeseries
    assert captured["source_mot"] == mot


def test_resume_rejects_player_folder_with_empty_video_shell(tmp_path: Path) -> None:
    player = tmp_path / "unknown_25"
    highlights = player / "highlights"
    highlights.mkdir(parents=True)
    for name in ("identity.yaml", "running.json", "heatmap.png"):
        (player / name).write_bytes(b"valid-placeholder")
    (highlights / "unknown_25_ev001.mp4").write_bytes(b"\x00" * 44)
    (highlights / "unknown_25_ev001.json").write_text("{}", encoding="utf-8")
    (player / "events_for_annotation.json").write_text(
        '{"total_events":1,"events":[{"video_file":"highlights/unknown_25_ev001.mp4"}]}',
        encoding="utf-8",
    )

    assert not formal_export.player_delivery_complete(player)

    (highlights / "unknown_25_ev001.mp4").write_bytes(b"\x00" * 2048)
    assert formal_export.player_delivery_complete(player)
