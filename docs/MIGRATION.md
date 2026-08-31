# 结构重构与迁移说明

来源：`Eeaker/football` @ `bfd54686703bd55a234b795c8123411bb30e55a2`。原仓库保持不变。

## 路径映射

| 旧路径/名称 | 当前路径/名称 | 原因 |
|---|---|---|
| `engine/tracking/week2lib` | `engine/tracking/tracking_lib` | 包名表达职责而不是研发阶段 |
| `engine/tracking/run_tracking_week2.py` | `engine/tracking/run_tracking.py` | 稳定 CLI 名称 |
| `botsort_week2_buffer.yaml` | `botsort_buffer.yaml` | 配置按功能命名 |
| `week2_run_metadata.json` | `tracking_run_metadata.json` | 运行产物脱离阶段号 |
| `event_index_week2.*` | `event_index.*` | 事件契约脱离阶段号 |
| `engine/week3_delivery` | `engine/match_analysis` | 模块职责是赛后分析，不是某周交付 |
| `week3lib` | `analysis_lib` | 包职责明确 |
| `run_week3.py` | `run_analysis.py` | 基础分析入口 |
| `run_week3_integrated.py` | `run_integrated_analysis.py` | 集成分析入口 |
| `demo_data/0724` | `demo_data/reference_match` | demo 按用途命名 |
| `demo_data/.../week3_integrated` | `demo_data/.../match_analysis` | 结果按类型命名 |

## 确认删除的重复/过时代码

1. `engine/football_metric_running_github`：与 `engine/football_metric_running` 的 Git tree SHA 完全一致，删除副本并把所有调用指向 canonical 目录。
2. 分析模块里复制的 actor / homography / team-feature：与 tracking 实现相同或只有一行“migrated”注释差异，改为通过 `analysis_lib/tracking_adapter.py` 复用。
3. 两个号码 OCR CLI：保留参数更完整的实现，统一命名为 `run_jersey_ocr.py`。
4. 旧 mock 报告脚本：标准报告脚本已经内置 mock payload 与真实输入兼容逻辑，因此删除旧实现。
5. `.playwright-cli` 快照、旧阶段报告和 `docs/source_notes`：属于调试/历史材料，不参与运行，移除以降低文档噪声。

## 兼容性说明

新仓库的**源码和新生成项目**只使用当前命名。若要搬迁旧 `runtime/projects`，应把旧分析输出目录映射到 `outputs/match_analysis`，并把旧追踪元数据/事件索引文件按上表重命名后再恢复；不要在新代码里长期保留两套路径分支。
