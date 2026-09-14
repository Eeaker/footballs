from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".venv", "__pycache__", ".git", ".pytest_cache"}
SKIP_SUFFIX = {".pyc", ".pyo"}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def distributable_files() -> list[Path]:
    out = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_PARTS for part in rel.parts) or p.suffix.lower() in SKIP_SUFFIX:
            continue
        if rel.parts[:2] == ("runtime", "projects") and len(rel.parts) >= 3 and rel.parts[2] != ".gitkeep":
            continue
        if rel.parts and rel.parts[0] == "runtime" and rel.parts[:2] != ("runtime", "projects"):
            continue
        out.append(p)
    return out


def main() -> None:
    audit = json.loads((ROOT / "CHAIN_AUDIT.json").read_text(encoding="utf-8"))
    files = distributable_files()
    manifest = {
        "schema_version": 4,
        "system": "赛场洞察 Football Insight",
        "version": "3.0.0",
        "generated_at": now(),
        "product_positioning": "面向场馆的容器化足球比赛分析系统；管理端跨平台，正式 BF16 分析运行于 NVIDIA 节点。",
        "formal_workflow": [
            "新建项目", "上传视频/名单与视频体检", "参数设置", "多锚点动态标定", "正式AI分析",
            "身份/传球/八维人工复核", "分层结果中心", "正式报告", "完整归档",
        ],
        "production_engine": {
            "tracking": True,
            "onboarding_video_health": True,
            "field_filter_and_global_ids": True,
            "identity_quality_audit": True,
            "identity_two_stage_recovery": True,
            "reid_full_parameter_bf16": True,
            "jersey_number_ocr": True,
            "multi_anchor_dynamic_calibration": True,
            "metric_running": True,
            "possession_and_passes": True,
            "events": True,
            "target_highlights": True,
            "dynamic_2d_replay": True,
            "player_cards": True,
            "formal_match_report": True,
            "source_sha256_audit": {
                "status": audit.get("status"),
                "matched": (audit.get("summary") or {}).get("critical_files_matched"),
                "total": (audit.get("summary") or {}).get("critical_files_total"),
            },
        },
        "calibration": {
            "mode": "dynamic_rotation_multi_anchor",
            "existing_config_upload": True,
            "manual_multi_anchor": True,
            "independent_scale_validation": True,
            "per_frame_homography": True,
            "coverage_gate": True,
            "vid_stride": 1,
        },
        "formal_features": {
            "persistent_projects": True,
            "video_upload_preview": True,
            "original_onboard_video_health": True,
            "roster_csv_json": True,
            "parameter_presets_and_advanced": True,
            "portable_parameter_template_import_export": True,
            "multi_anchor_dynamic_calibration": True,
            "uploaded_dynamic_calibration": True,
            "dynamic_calibration_download_reuse": True,
            "preflight_gate": True,
            "retry_from_failed_stage": True,
            "run_history_and_logs": True,
            "gpu_concurrency_guard": True,
            "model_upload_from_ui": True,
            "windows_default_model_downloader": True,
            "dynamic_2d_replay": True,
            "target_highlights": True,
            "player_cards": True,
            "pass_human_review": True,
            "technical_id_to_real_player_confirmation": True,
            "unmerged_identity_quarantine": True,
            "clear_frame_player_avatar": True,
            "ocr_team_jersey_display_id": True,
            "human_eight_dimension_assessment": True,
            "team_semantic_mapping": True,
            "html_report_print_to_pdf": True,
            "artifact_manifest_sha256": True,
            "project_result_zip": True,
        },
        "deployment": {
            "primary_target": "Windows one-click native runtime; Docker optional for venue Postgres/Redis/MinIO",
            "host_python_required": False,
            "one_click_windows": "DEPLOY_ONE_CLICK_WINDOWS.bat",
            "one_click_posix": "deploy.sh",
            "docker_windows": "DEPLOY_DOCKER_WINDOWS.bat",
            "metadata_database": "PostgreSQL 16 / hash partitioned",
            "coordination": "Redis 7 distributed leases",
            "artifact_storage": "MinIO S3-compatible objects",
            "workspace_policy": "tmpfs staging only; not durable storage",
            "compute_policy": "CUDA BF16 only for ReID training and inference",
            "default_model_installer": "automatic container startup download or UI upload",
            "required_model": "models/yolov8x.pt",
            "default_max_concurrent_gpu_jobs": 1,
        },
        "verification": {
            "python_compile": True,
            "javascript_syntax": True,
            "formal_product_api_workflow": "PASS",
            "human_eight_dimension_assessment": "PASS",
            "portable_configuration_reuse": "PASS",
            "multi_anchor_dynamic_calibration_smoke": "PASS (workflow smoke only, not a metric-accuracy claim)",
            "package_and_product_ready": True,
            "fresh_new_video_gpu_end_to_end_verified": False,
            "fresh_inference_note": "Must be recorded on a BF16-capable NVIDIA deployment node with yolov8x.pt installed; demo results do not count.",
        },
        "docs": [
            "README.md", "docs/README.md", "docs/USER_GUIDE.md", "docs/DEPLOYMENT.md",
            "docs/OPERATIONS.md", "docs/ARCHITECTURE.md", "docs/DEVELOPMENT.md",
            "docs/ACCEPTANCE.md", "docs/MIGRATION.md", "CHAIN_AUDIT.json",
        ],
        "package_contents": {
            "top_level_folders": ["app", "engine", "deploy", "models", "scripts", "docs", "examples"],
            "distributable_file_count": len(files),
            "distributable_uncompressed_bytes": sum(p.stat().st_size for p in files),
            "not_bundled_by_design": [
                "Docker runtime", "NVIDIA driver/toolkit", "yolov8x.pt when startup download is unavailable",
                "user full-match videos", "historical research outputs unrelated to runtime",
            ],
        },
    }
    (ROOT / "SYSTEM_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    runtime_report = ROOT / "runtime" / "acceptance_report.json"
    if runtime_report.is_file():
        report = json.loads(runtime_report.read_text(encoding="utf-8"))
        report["release_metadata_generated_at"] = now()
        (ROOT / "BUILD_ACCEPTANCE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest files={len(files)} bytes={sum(p.stat().st_size for p in files)}")


if __name__ == "__main__":
    main()
