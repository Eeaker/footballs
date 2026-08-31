from __future__ import annotations

from pathlib import Path
import sys


# Keep tests runnable both from match_analysis/ and from the shared repository root.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
