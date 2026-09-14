from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runtime" / "acceptance_report.json"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def check(name: str, ok: bool, detail: str = "", category: str = "package") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail, "category": category}


def main() -> int:
    rows: list[dict] = []
    required_files = [
        "DEPLOY_ONE_CLICK_WINDOWS.bat", "DEPLOY_DOCKER_WINDOWS.bat", "deploy.ps1", "deploy.sh", "deploy/Dockerfile", "deploy/compose.yaml", "deploy/compose.gpu.yaml",
        "deploy/postgres/init/001_schema.sql", "app/features/projects/repository.py", "app/features/jobs/coordination.py",
        "app/features/artifacts/store.py", "app/features/analytics/importer.py", "app/features/identity/policy.py", "app/features/identity/avatar.py",
        "app/app.py", "app/static/index.html", "app/static/app.js", "app/static/styles.css", "app/services/video_health.py",
        "engine/tracking/run_pipeline.py", "engine/tracking/onboard/video_health.py",
        "engine/football_metric_running/src/running_metrics_v1/build_multi_anchor_dynamic_calibration.py",
        "engine/match_analysis/run_integrated_analysis.py", "engine/match_analysis/run_jersey_ocr.py",
        "engine/match_analysis/generate_player_card.py",
        "engine/identity_audit/mode_split/audit_mot.py",
        "engine/identity_resolution/scripts/run_full_match.py", "engine/identity_resolution/src/second_stage.py",
        "engine/identity_resolution/src/features.py", "engine/identity_resolution/reid_model/train_adapt.py",
        "CHAIN_AUDIT.json", "CHECK_WINDOWS.bat",
        "DOWNLOAD_MODEL_WINDOWS.bat", "REPAIR_WINDOWS.bat", "DIAGNOSE_WINDOWS.bat", "PRESENT_WINDOWS.vbs", "STOP_WINDOWS.bat", "PREPARE_OFFLINE_WINDOWS.bat", "INSTALL_OFFLINE_WINDOWS.bat", "scripts/windows_prepare_offline.ps1", "scripts/windows_install_offline.ps1", "scripts/windows_torch_probe.py", "scripts/install_default_model.py", "scripts/windows_install.ps1",
        "docs/USER_GUIDE.md", "docs/DEPLOYMENT.md", "docs/OPERATIONS.md", "docs/ARCHITECTURE.md", "docs/DEVELOPMENT.md", "docs/MIGRATION.md",
    ]
    for rel in required_files:
        p = ROOT / rel
        rows.append(check(rel, p.is_file(), "ready" if p.is_file() else "missing"))

    launcher_proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_windows_launchers.py")], cwd=ROOT, capture_output=True, text=True)
    rows.append(check("Windows launcher encoding", launcher_proc.returncode == 0, (launcher_proc.stdout + launcher_proc.stderr)[-4000:], "package"))

    audit_proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_engine_chain.py")], cwd=ROOT, capture_output=True, text=True)
    rows.append(check("engine source SHA256 audit", audit_proc.returncode == 0, (audit_proc.stdout + audit_proc.stderr)[-4000:], "engine"))

    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_product.py")], cwd=ROOT, capture_output=True, text=True)
    rows.append(check("formal product workflow", proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:], "product"))

    deps = {name: importlib.util.find_spec(name) is not None for name in [
        "fastapi", "uvicorn", "multipart", "cv2", "numpy", "scipy", "yaml", "torch", "torchvision",
        "ultralytics", "easyocr", "skimage", "shapely", "openpyxl",
    ]}
    inference_dependencies = all(deps.values())
    rows.append(check("inference python dependencies", inference_dependencies, ", ".join(k for k, v in deps.items() if not v) or "all installed", "environment"))
    model = ROOT / "models" / "yolov8x.pt"
    rows.append(check("YOLO weights", model.is_file(), str(model) if model.is_file() else "not bundled; upload in System Status", "environment"))

    product_ok = all(r["ok"] for r in rows if r["category"] in {"package", "engine", "product"})
    inference_ready = all(r["ok"] for r in rows if r["category"] == "environment")
    payload = {
        "schema_version": 2,
        "system": "Football Insight",
        "version": "3.0.0",
        "generated_at": now(),
        "package_and_product_ready": product_ok,
        "fresh_inference_environment_ready": inference_ready,
        "fresh_inference_verified": False,
        "note": "fresh_inference_verified remains false until a new full match is run end-to-end on a BF16-capable NVIDIA deployment node.",
        "checks": rows,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Football Insight V3.0.0 acceptance")
    print("=" * 72)
    for row in rows:
        tag = "PASS" if row["ok"] else ("WAIT" if row["category"] == "environment" else "FAIL")
        print(f"{tag:<5} [{row['category']:<11}] {row['name']}")
    print("=" * 72)
    print("系统源码/产品包：", "READY" if product_ok else "NOT READY")
    print("当前机器新视频推理：", "READY" if inference_ready else "WAITING FOR ENVIRONMENT/MODEL")
    print("目标 BF16 NVIDIA 节点新素材端到端验收：NOT YET RECORDED")
    print("报告：", REPORT)
    return 0 if product_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
