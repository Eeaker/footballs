from pathlib import Path

from app.features.identity.policy import build_display_id, human_confirmed_groups, load_identity_catalog
from engine.identity_resolution.src.features import require_bf16
import pytest


def test_catalog_only_publishes_confirmed_multi_id_groups(tmp_path: Path):
    root = tmp_path / "identity_resolution" / "filtered"
    root.mkdir(parents=True)
    (root / "identity_catalog.json").write_text(
        '{"published_players":[{"canonical_id":7,"source_ids":[7,51,161],"confirmed":true},'
        '{"canonical_id":9,"source_ids":[9],"confirmed":true},'
        '{"canonical_id":27,"source_ids":[27,120],"confirmed":false}],'
        '"isolated_ids":[9,27],"quarantine_ids":[114]}', encoding="utf-8"
    )
    catalog = load_identity_catalog(tmp_path)
    assert catalog.published == {7: (7, 51, 161)}
    assert catalog.isolated_ids == frozenset({9, 27})
    assert catalog.quarantine_ids == frozenset({114})


def test_human_confirmation_requires_an_actual_merge():
    groups = human_confirmed_groups({
        "7": {"person_key": "linked:7,51", "linked_global_ids": [7, 51]},
        "51": {"person_key": "linked:7,51", "linked_global_ids": [7, 51]},
        "9": {"person_key": "manual:blue|25|a", "linked_global_ids": [9]},
    })
    assert groups == {7: (7, 51)}


def test_display_id_uses_team_and_ocr_number():
    assert build_display_id("蓝队", "25", "ID 7") == "蓝队25号"
    assert build_display_id("蓝队", "待确认", "ID 7") == "ID 7"


def test_reid_never_silently_falls_back_to_cpu_precision():
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        require_bf16("cpu")
