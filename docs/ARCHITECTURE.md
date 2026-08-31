# 系统架构

## 1. 分层

```text
浏览器 / app/static
        │
        ▼
FastAPI / app
  项目、上传、配置、标定、任务、复核、报告、导出
        │
        ▼
app/services/pipeline.py
        │
        ├─ engine/tracking
        ├─ engine/football_metric_running
        ├─ engine/match_analysis
        └─ engine/identity_audit
        │
        ▼
runtime/projects/<project_id>
```

## 2. 引擎职责

| 模块 | 责任 | 主要输出 |
|---|---|---|
| `tracking` | 人/球检测、追踪、场内过滤、技术 ID、候选事件 | MOT、球轨迹、追踪视频、事件索引 |
| `football_metric_running` | 动态 Homography、像素到米制、跑动/速度 | 逐帧米制坐标、跑动汇总 |
| `match_analysis` | OCR、队伍、球权、传球候选、2D 球场、球员卡 | 号码、球权、传球、卡片、报告输入 |
| `identity_audit` | 外观模式与技术 ID 切换质检 | 可疑候选，不直接改写 MOT |

`match_analysis` 通过 `analysis_lib/tracking_adapter.py` 复用追踪模块的 actor / homography / team-feature 实现，避免维护多份同源代码。

## 3. 正式项目目录

```text
runtime/projects/<project_id>/
├─ input/
├─ calibration/
├─ config/
├─ logs/
├─ outputs/
│  ├─ tracking/
│  ├─ number_ocr/
│  ├─ match_analysis/
│  │  ├─ analysis/
│  │  └─ metric_running/
│  ├─ player_cards/
│  ├─ highlights/
│  └─ match_report.html
├─ reviews/
└─ exports/
```

## 4. 关键数据契约

系统区分三类语义：

- **机器事实**：检测框、MOT、球位置、米制坐标等直接算法产物；
- **机器候选**：号码、传球、关键事件、身份污染等需要复核的高层判断；
- **人工确认**：真实姓名/号码、传球标签、队伍语义、八维评分。

人工确认会覆盖展示层的语义，但不会无痕改写原始机器产物。

追踪元数据统一使用 `tracking_run_metadata.json`；事件索引统一使用 `event_index.json`/`event_index.csv`。阶段号不再进入稳定数据契约。

## 5. 动态标定门控

拍摄点固定但相机旋转时，系统通过多个锚点生成逐帧 Homography，并检查视频元数据、独立尺度误差和有效帧覆盖率。标定不通过时，米制跑动、球权和传球链路停止，而不是继续输出伪精确数据。

## 6. 调度与恢复

`app/services/pipeline.py` 是产品层唯一编排入口。每个阶段开始前清理该阶段的残留产物；恢复时先验证前置阶段关键文件，避免在损坏结果上继续计算。GPU 并发默认限制为 1，可通过环境变量调整。
