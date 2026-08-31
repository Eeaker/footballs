from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".venv", "__pycache__", ".git", ".pytest_cache"}
SKIP_SUFFIX = {".pyc", ".pyo"}


def include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in SKIP_PARTS for part in rel.parts):
        return False
    if path.suffix.lower() in SKIP_SUFFIX:
        return False
    # Runtime is intentionally shipped empty; app recreates the demo project from demo_data.
    if len(rel.parts) >= 2 and rel.parts[0] == "runtime" and rel.parts[1] != "projects":
        return False
    if len(rel.parts) >= 3 and rel.parts[0] == "runtime" and rel.parts[1] == "projects" and rel.parts[2] != ".gitkeep":
        return False
    return True


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT.parent / "football_insight_system_v2_3_3_windows_full.zip"))
    args = ap.parse_args()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    prefix = ROOT.name
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and include(p))
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            arc = Path(prefix) / path.relative_to(ROOT)
            zf.write(path, arcname=str(arc))
    print(out)
    print(f"files={len(files)} size={out.stat().st_size} sha256={sha256(out)}")


if __name__ == "__main__":
    main()
