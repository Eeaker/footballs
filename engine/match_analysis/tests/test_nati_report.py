import json
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator

import render_nati_standard_report as report


ROOT = Path(__file__).resolve().parents[1]


def _pending_payload():
    payload = report._empty_payload()
    payload["player"] = {
        "player_id": "white_26",
        "team": "white",
        "jersey_number": 26,
        "age_group": "U12",
    }
    payload["key_metrics"] = [
        {"key": "total_distance_m", "value": 596.884, "label": "跑动距离"},
        {"key": "sprint_count", "value": 100, "label": "冲刺次数"},
        {"key": "max_speed_ms", "value": 2.022, "label": "最高速度"},
    ]
    return payload


def test_pending_payload_preserves_facts_without_mock_scores():
    normalized = report.normalize_payload(_pending_payload())

    assert normalized["schema_version"] == "nati-assessment-input-v1"
    assert [metric["display"] for metric in normalized["key_metrics"]] == ["596.9 m", "100", "2.02 m/s"]
    assert normalized["overall"]["ca_score"] is None
    assert set(normalized["radar"]["dimensions"].values()) == {None}
    assert normalized["style_archetype"]["reference_player"] == "待评估"
    assert all(row["fit"] is None for row in normalized["position_recommendations"])


def test_pending_payload_builds_html_with_explicit_pending_state():
    html = report.build_html(report.normalize_payload(_pending_payload()))

    assert "596.9 m" in html
    assert "待评估" in html
    assert "佩德里" not in html
    assert "向 A 进发" not in html


def test_v2_input_is_adapted_to_official_v1_contract():
    payload = deepcopy(report.mock_payload())
    payload["schema_version"] = "nati-assessment-input-v2"
    normalized = report.normalize_payload(payload)

    assert normalized["schema_version"] == "nati-assessment-input-v1"
    assert normalized["source_schema_version"] == "nati-assessment-input-v2"


def test_mock_and_pending_outputs_validate_against_schema():
    schema = json.loads((ROOT / "assessment_report_input.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    assert list(validator.iter_errors(report.normalize_payload(report.mock_payload()))) == []
    assert list(validator.iter_errors(report.normalize_payload(_pending_payload()))) == []
