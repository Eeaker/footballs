# 产物清单与大文件策略

更新时间：2026-08-09。

## 仓库内核心输出

### 主流水线

- `artifacts/pipeline/tracking_mot.txt`：第二版统一帧域完整 MOT，覆盖 1–62,204 帧。
- `artifacts/pipeline/event_index.json` / `.csv`：182 个事件的结构化索引。
- `artifacts/pipeline/comparison_outputs_vs_v2_5.csv`：5 个事件新旧时间戳对比。
- `artifacts/pipeline/full_rerun_v2_log.txt`：第二版全量复跑日志。
- `artifacts/pipeline/sample_clips/*.mp4`：5 个代表性事件切片，覆盖关键动作、传球/解围
  和射门；完整 20 个切片未重复复制。

### 米制跑动

- `artifacts/metric_running/calibration/*`：51 s 四点标定输入、H、验证和参考图。
- `artifacts/metric_running/full_run/dynamic_calibration_45x25.json`：62,204 帧动态 H。
- `artifacts/metric_running/full_run/metrics/player_running_summary.csv`：35 个 tracking ID 汇总。
- `artifacts/metric_running/full_run/metrics/player_running_timeseries.csv`：逐帧米制坐标、
  位移和速度。
- `artifacts/metric_running/full_run/metrics/running_quality_report.json`：碰撞、越界、短段等。
- `artifacts/metric_running/full_run/qa_*.jpg`：1、18、30 分钟抽检截图。
- `artifacts/metric_running/full_run/validation_left_center_right.*`：左右旋转三视角验证。
- `artifacts/metric_running/demo/*`：3 s 固定 H Demo 和小型对照结果。

## Git LFS 文件

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `large_artifacts/pipeline/tracking_vis_720p_review.mp4` | 775232582 | `AC3FE603E301F2CD3E6BDC3D41630A3F48C2E1B6F320DD9FD94DAD25925BC6D6` |
| `large_artifacts/metric_running/full_running_dynamic_45x25_720p_review.mp4` | 826046287 | `1AE1049CB88D91CD29A014B1135827E501F972B55F156F3B15711C490B70150E` |

此外，约 55.6 MiB 的
`artifacts/metric_running/full_run/metrics/player_running_timeseries.csv` 也配置为
Git LFS，以避免普通 Git 对超过 50 MiB 文件的警告和历史膨胀。所有 MP4（包括事件
样片）统一由 LFS 管理。

## 未复制到仓库的大文件

这些文件保留在原工作区，因隐私、版权或 GitHub 单文件限制未复制：

| 文件 | 字节 | SHA-256 | 原工作区路径 |
|---|---:|---|---|
| 原始比赛视频 | 1791973968 | `A6CAC4651B1A98EF77365A6CC886AC62842ED1DE52BF411E9845E7FD5A5322C4` | `C:\Users\23159\Downloads\football\足球视频7月24日.mp4` |
| YOLOv8x 权重 | 136890692 | `3DF4ADA6B4DAD6D657868F2FDF7FAECFB34DCFCCF3A25C4B82079064718524C8` | `C:\Users\23159\Downloads\football\yolov8x.pt` |
| 1080p 跟踪可视化 | 3829735618 | `842C7EA0C417294BF54A541A3053F089D786DBA75695C6149F5759ADAB90DEE5` | `run_artifacts/frame_domain_fix_20260806/full_rerun_outputs_4090_v2/tracking_vis.mp4` |
| 1080p 米制跑动成片 | 2643310613 | `E85C3B9B3187D700CFDBAE0EAA371391AB874FCDB2AD8C256E9BBC5872DE111B` | `running_metrics_v1/results/full_rotation_run/full_running_dynamic_45x25.mp4` |

上述两个 1080p 输出均超过 GitHub Free/Pro 当前 2 GB Git LFS 单文件上限。不要尝试
直接加入提交历史；即使之后删除，巨型对象仍可能留在 Git 历史中。

## 公开前检查

1. 确认未成年人比赛画面是否允许公开。
2. 确认模型权重许可及是否需要只提供官方下载链接。
3. 选择项目级许可证。
4. 运行 `scripts/verify_release.ps1`。
5. 使用 `git lfs ls-files` 确认所有 MP4 都由 LFS 跟踪。
