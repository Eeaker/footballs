from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "CHAIN_AUDIT.json"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    if not AUDIT.is_file():
        print("FAIL  CHAIN_AUDIT.json missing")
        return 2
    data = json.loads(AUDIT.read_text(encoding="utf-8"))
    failed = []
    rows = data.get("critical_files", [])
    print("Football Insight first-party source integrity audit")
    print("=" * 76)
    for item in rows:
        rel = item.get("path") or item.get("system_path")
        expected = item.get("sha256") or item.get("source_sha256")
        path = ROOT / rel
        actual = sha256(path) if path.is_file() else None
        ok = actual == expected
        if not ok:
            failed.append(rel)
        print(f"{'PASS' if ok else 'FAIL':<5} {rel}")
    print("=" * 76)
    print(f"critical files: {len(rows)-len(failed)}/{len(rows)}")
    if failed:
        for rel in failed: print(" -", rel)
        return 1
    print("integrity audit: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
