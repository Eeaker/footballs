from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "CHAIN_AUDIT.json"
V3_CRITICAL_FILES = [
    "app/features/projects/repository.py",
    "app/features/jobs/coordination.py",
    "app/features/artifacts/store.py",
    "app/features/analytics/importer.py",
    "app/features/identity/policy.py",
    "app/features/identity/avatar.py",
    "engine/identity_resolution/scripts/run_full_match.py",
    "engine/identity_resolution/src/features.py",
    "engine/identity_resolution/src/second_stage.py",
    "engine/identity_resolution/reid_model/train_adapt.py",
    "deploy/compose.yaml",
    "deploy/compose.gpu.yaml",
    "deploy/Dockerfile",
    "deploy/postgres/init/001_schema.sql",
    "DEPLOY_ONE_CLICK_WINDOWS.bat",
    "deploy.ps1",
    "deploy.sh",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser(description="Verify or refresh the reviewed source integrity manifest")
    parser.add_argument("--update", action="store_true", help="record current reviewed source hashes")
    args = parser.parse_args()
    if not AUDIT.is_file():
        print("FAIL  CHAIN_AUDIT.json missing")
        return 2
    data = json.loads(AUDIT.read_text(encoding="utf-8"))
    if args.update:
        indexed = {(item.get("path") or item.get("system_path")): item for item in data.get("critical_files", [])}
        for rel in V3_CRITICAL_FILES:
            indexed.setdefault(rel, {"path": rel, "role": "venue_v3"})
        updated = []
        for rel, item in sorted(indexed.items()):
            path = ROOT / rel
            if path.is_file():
                item["path"] = rel
                item["sha256"] = sha256(path)
                item.pop("source_sha256", None)
                updated.append(item)
        data["critical_files"] = updated
        data["summary"] = {"critical_files_total": len(updated), "critical_files_matched": len(updated)}
        data["generated_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        data["status"] = "verified"
        AUDIT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"updated {len(updated)} reviewed hashes")
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
