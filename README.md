# Football Insight · 足球比赛视频分析系统

Football Insight 是一个面向 **Windows 单机 / NVIDIA GPU** 的足球比赛视频分析系统。它把视频体检、人物/足球追踪、动态标定、号码识别、米制跑动、球权与传球候选、人工复核、2D 回放、球员卡和正式报告串成一条可落地的工作流。

> 当前功能版本保持为 **2.3.3**。本仓库在不改变核心业务口径的前提下重构了目录、公共代码和文档。源仓库 `Eeaker/football` 不做修改；迁移来源 commit：`bfd54686703bd55a234b795c8123411bb30e55a2`。

## 核心工作流

```text
新建项目
  → 上传视频 / 球员名单
  → 视频健康检查与参数配置
  → 多锚点动态标定
  → 人 / 球追踪与技术 ID
  → 队伍提示与多帧号码识别
  → 米制跑动 / 球权 / 传球候选
  → 人工复核与真实身份确认
  → 2D 回放 / 高光 / 球员卡
  → 报告与完整归档
```

产品界面保持四个阶段：**追踪中 → 号码识别 → 事件检测 → 报告生成**。

## 快速开始（Windows）

推荐：Windows 10/11 x64、64 位 Python 3.11/3.12、NVIDIA GPU。

1. 首次运行双击 `RUN_WINDOWS.bat`；它会在需要时创建 `.venv` 并安装依赖。
2. 若缺少检测模型，双击 `DOWNLOAD_MODEL_WINDOWS.bat`，或在系统状态页上传 `yolov8x.pt`。
3. 运行 `CHECK_WINDOWS.bat` 检查启动器、依赖、代码完整性和产品 API。
4. 浏览器打开后创建项目，上传真实比赛视频并完成动态标定。

更完整的环境、GPU、离线安装和故障排查见 [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)。

## 仓库结构

```text
app/                       FastAPI 产品层、静态前端与项目服务
engine/
  tracking/                检测、追踪、场内过滤、技术 ID、候选事件
  football_metric_running/ 动态 Homography、米制坐标、跑动与速度
  match_analysis/          号码识别、球权/传球、2D 球场、球员卡、报告数据
  identity_audit/          技术 ID 外观污染/切换候选审计
  repository_qa/           仓库级验证入口
demo_data/reference_match/ 只读参考比赛结果，用于产品层验收
scripts/                   安装、检查、发布与产品验收脚本
docs/                      用户、部署、架构、开发、运维与验收文档
models/                    本地模型目录（权重默认不入库）
runtime/                   运行时项目与日志（自动创建，不入库）
```

## 这次重构解决了什么

- 删除 `engine/football_metric_running_github`：它与 `engine/football_metric_running` 的整棵 Git tree 完全相同。
- 删除分析模块中复制的 tracking actor / homography / team-feature 实现，统一通过 `analysis_lib/tracking_adapter.py` 复用追踪模块。
- 合并重复的号码识别 CLI，只保留 `run_jersey_ocr.py`。
- 删除已被标准报告完全覆盖的旧 mock 报告脚本。
- 清理 Playwright 临时快照、旧阶段报告和重复历史说明。
- 所有阶段/日期式目录和文件名改为功能命名；运行产物也使用 `match_analysis/`、`tracking_run_metadata.json`、`event_index.json` 等稳定名称。
- 修复仓库声称存在但被 `.gitignore` 排除的 `PRESENT_WINDOWS.vbs`。

详细映射见 [`docs/MIGRATION.md`](docs/MIGRATION.md)。

## 开发与验证

```bash
python -m compileall -q app engine scripts
python -m pytest -q app/tests
(cd engine/tracking && python -m pytest -q)
(cd engine/football_metric_running && PYTHONPATH=src python -m pytest -q)
(cd engine/identity_audit && PYTHONPATH=../tracking:. python -m pytest -q)
(cd engine/match_analysis && PYTHONPATH=. python -m pytest -q)
python scripts/audit_engine_chain.py
python scripts/verify_product.py
```

完整 AI 推理还需要 `requirements-ai.txt` 和模型权重。仓库级测试通过不等价于“目标 Windows GPU 上的新比赛已完成全链路推理”，后者必须在目标机器用真实新素材单独验收。

## 文档导航

- [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)：产品使用流程
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)：Windows / GPU / 离线部署
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)：模块、数据流与关键契约
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)：开发约定、命名、测试和去重原则
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md)：运行、备份、恢复与排错
- [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md)：发布与真实推理验收
- [`docs/MIGRATION.md`](docs/MIGRATION.md)：旧结构到当前结构的迁移说明

## 数据与模型边界

仓库不分发 NVIDIA 驱动、CUDA、Python 虚拟环境、完整比赛视频或 `yolov8x.pt`。`demo_data/reference_match` 只用于界面和数据契约验收，不应被当作新素材推理证明。号码、传球、真实身份和八维评估在未人工确认时仍按候选/待评估展示。
