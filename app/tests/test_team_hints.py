from pathlib import Path

from app.services import team_hints


def test_team_hints_imports_match_analysis_from_bundled_engine():
    expected = Path(__file__).resolve().parents[2] / "engine" / "match_analysis"
    assert team_hints.MATCH_ANALYSIS_ROOT == expected
    assert (team_hints.MATCH_ANALYSIS_ROOT / "analysis_lib" / "io.py").is_file()

