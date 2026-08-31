from __future__ import annotations

from datetime import timedelta
import json

from openpyxl import Workbook

from import_annotation_xlsx import convert


def test_annotation_ledger_is_converted_with_provisional_identity(tmp_path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "标注台账"
    sheet.append(["标题"])
    headers = ["锚点", "锚点时间", "切片ID", "事件类型提示", "观察对象", "起始时间戳",
               "结束时间戳", "时长（秒）", "主维度", "行为标签（1～3个）", "拿不准",
               "实际耗时（分钟）", "备注", "切片文件"]
    sheet.append(headers)
    sheet.append(["A01", timedelta(seconds=10), "ev001", "快速传球", "黄方",
                  timedelta(seconds=7), timedelta(seconds=13), 6, "决", "一脚出球", "否", 3,
                  "方向明确", "ev001.mp4"])
    xlsx = tmp_path / "events.xlsx"
    workbook.save(xlsx)
    detector = tmp_path / "detector.json"
    detector.write_text(json.dumps([{
        "event_time_seconds": 10.5, "primary_global_id": 3, "score": 80,
        "actor_candidates": [{"global_id": 3}, {"global_id": 4}],
    }]), encoding="utf-8")
    numbers = tmp_path / "numbers.json"
    numbers.write_text(json.dumps({
        "eligible_confirmed": [{"global_id": 3, "team": "yellow", "final_number": 15, "confidence": .9}],
    }), encoding="utf-8")
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "ev001.mp4").write_bytes(b"clip")
    output = tmp_path / "events.json"
    payload = convert(xlsx=xlsx, detector_events=detector, clips_dir=clips,
                      numbers=numbers, output=output, max_match_seconds=8)
    event = payload["events"][0]
    assert event["event_type"] == "pass_or_distribution"
    assert event["player_id"] == "yellow_15"
    assert event["secondary_global_id"] == 4
    assert event["actor_assignment_status"] == "provisional_nearest_detector_event"
