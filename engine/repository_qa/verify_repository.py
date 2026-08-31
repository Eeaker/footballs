from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Check:
    name: str
    cwd: Path
    args: tuple[str, ...]


CHECKS = (
    Check("tracking frame-domain", ROOT, ("-m", "pytest", "-q", "tests/test_frame_domain.py")),
    Check("tracking and new-video onboarding", ROOT / "tracking", ("-m", "pytest", "-q")),
    Check("metric running", ROOT / "football_metric_running", ("-m", "pytest", "-q")),
    Check("final delivery", ROOT / "match_analysis", ("-m", "pytest", "-q")),
    Check("identity team-switch audit", ROOT / "tools" / "identity_team_audit", ("-m", "pytest", "-q")),
    Check("new-video onboarding CLI", ROOT / "tracking", ("onboard.py", "--help")),
    Check("tracking CLI", ROOT / "tracking", ("run_pipeline.py", "--help")),
    Check("player-card CLI", ROOT / "match_analysis", ("generate_player_card.py", "--help")),
    Check("task3 CLI", ROOT / "match_analysis", ("run_analysis.py", "--help")),
)


def main() -> int:
    failures: list[str] = []
    environment = os.environ.copy()
    running_src = str(ROOT / "football_metric_running" / "src")
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (running_src, existing_pythonpath) if item
    )
    for check in CHECKS:
        print(f"\n=== {check.name} ===", flush=True)
        result = subprocess.run(
            (sys.executable, *check.args), cwd=check.cwd, env=environment
        )
        if result.returncode:
            failures.append(check.name)
    if failures:
        print("\nFAILED: " + ", ".join(failures))
        return 1
    print("\nALL CHECKS PASSED (144 tests + 4 CLI smoke checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
