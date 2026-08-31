from __future__ import annotations

from pathlib import Path
import json

from .models import CalibrationResult, MotionHealth, TeamColorResult, TrialMetrics


def write_report(output: str | Path, health: MotionHealth, colors: TeamColorResult | None,
                 calibration: CalibrationResult, baseline: TrialMetrics | None,
                 candidate: TrialMetrics | None, selected: str, reason: str) -> Path:
    """输出适配报告 Markdown，并在同目录保存机器可读 JSON。"""
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    team_text = "未执行" if colors is None else (
        f"K2={colors.silhouette_k2:.3f}，K3={colors.silhouette_k3:.3f}，建议/确认 K="
        f"{colors.recommended_k}/{colors.selected_k}（固定自动三簇），候选轨迹原型 {colors.person_crops} 个")
    calibration_text = (f"模式 {calibration.mode}；关键帧 {len(calibration.keyframes)}；"
                        f"验证={'通过' if calibration.validated else '未通过或未执行'}")
    def metric_text(item):
        return "未执行" if item is None else (f"平均框/帧 {item.mean_boxes_per_frame}，局部ID {item.local_id_total}，"
            f"新ID/分钟 {item.new_ids_per_minute}，轨迹中位长度 {item.median_track_length_frames} 帧")
    text = f"""# 新视频上机适配报告

## Stage A 视频体检

- 视频：`{health.metadata.path}`
- {health.metadata.width}×{health.metadata.height}，{health.metadata.fps:.3f} fps，{health.metadata.duration_seconds:.1f} 秒
- 相机类型：`{health.motion_type}`；动态 H 可用：`{health.dynamic_h_usable}`
- 位移中位/P75/P90：{health.median_translation_px:.2f}/{health.p75_translation_px:.2f}/{health.p90_translation_px:.2f}px；运动帧对比例：{health.moving_pair_ratio:.1%}
- 建议：{health.recommendation}

## Stage B/C 场内轨迹与队色

{team_text}

## Stage D 米制标定

{calibration_text}。独立线段误差阈值为 {calibration.validation_threshold_m:.2f} 米。

## Stage E 参数试跑

- 基线：{metric_text(baseline)}
- 候选：{metric_text(candidate)}
- 推荐：`{selected}`。{reason}

## 使用说明

主流程用 `python run_tracking.py --config <本目录配置文件>` 加载。显式命令行参数会覆盖 YAML。米制结果只有在标定验证通过时启用；转动镜头不得将首帧静态 H 套到全片。
"""
    target.write_text(text, encoding="utf-8")
    payload = {"health": health.to_dict(), "team_colors": colors.to_dict() if colors else None,
               "calibration": calibration.to_dict(), "baseline": baseline.to_dict() if baseline else None,
               "candidate": candidate.to_dict() if candidate else None, "selected": selected, "reason": reason}
    target.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
