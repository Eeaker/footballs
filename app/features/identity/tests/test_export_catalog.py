import importlib.util
import json
import pickle
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[4] / "engine" / "identity_resolution" / "scripts" / "run_full_match.py"
SPEC = importlib.util.spec_from_file_location("identity_full_match_export", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _inputs(tmp_path: Path):
    rows = [
        (0, 1, 0.0, 0.0, 20.0, 50.0, 0.9),
        (2, 2, 1.0, 0.0, 20.0, 50.0, 0.9),
        (0, 3, 50.0, 0.0, 20.0, 50.0, 0.9),
    ]
    audit = {"rows": rows, "nodes": [
        {"output_id": 1, "quarantine": False, "rows": [rows[0]]},
        {"output_id": 2, "quarantine": False, "rows": [rows[1]]},
        {"output_id": 3, "quarantine": False, "rows": [rows[2]]},
    ]}
    audit_path = tmp_path / "audit.pkl"
    audit_path.write_bytes(pickle.dumps(audit))
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps({"1": 1, "2": 1, "3": 3}), encoding="utf-8")
    return audit_path, mapping_path


def test_unmerged_identity_is_isolated_not_published(tmp_path: Path):
    audit, mapping = _inputs(tmp_path)
    stats = MODULE.export_final_identities(audit, mapping, tmp_path / "mot.txt", min_frames=1)
    assert stats["published_players"] == [{"canonical_id": 1, "source_ids": [1, 2], "confirmed": True}]
    assert stats["isolated_ids"] == [3]


def test_review_required_edge_is_not_a_player(tmp_path: Path):
    audit, mapping = _inputs(tmp_path)
    edges = tmp_path / "edges.json"
    edges.write_text('[{"from_oid":2,"to_oid":1,"requires_review":true}]', encoding="utf-8")
    stats = MODULE.export_final_identities(audit, mapping, tmp_path / "mot.txt", min_frames=1, accepted_edges_path=edges)
    assert stats["published_players"] == []
    assert stats["isolated_ids"] == [1, 2, 3]

