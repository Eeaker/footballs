# NATI · 足球比赛视频分析系统

Football Insight 是哪踢科技视觉分析链路的正式产品化实现，面向 Windows 10/11 x64 与 NVIDIA GPU 单机部署。系统覆盖比赛视频接入、动态标定、人/球追踪、技术 ID、号码识别、米制跑动、球权与传球候选、人工复核、2D 回放、高光、球员卡与正式报告。

## 一键部署

新 Windows 电脑只双击 **`DEPLOY_ONE_CLICK_WINDOWS.bat`**。不要先装 Docker、Python 或 pip，也不要再点别的安装脚本。

首次联网会从零检查并自动完成：下载独立 Python 3.12 到 `runtime/python`（本机没有 pip 时用 `ensurepip` 自举）、安装 VC++ 运行库与 AI 依赖、下载默认检测模型、创建桌面快捷方式，然后打开网页。有 NVIDIA 则走 GPU，装不上 CUDA 时自动回退 CPU。之后再点同一个文件只会启动。

场馆 Docker（PostgreSQL / Redis / MinIO）仍可用 `DEPLOY_DOCKER_WINDOWS.bat` 或 Linux/macOS 的 `./deploy.sh`。正式 ReID 全参微调与推理统一使用 BF16；无 GPU 时网页仍可启动，分析前会给出检查结果。

> 本仓库只交付正式源码、配置模板、测试与运维文档。比赛视频、球员数据、追踪结果、事件切片、模型权重、密钥及 Demo/参考比赛数据均不进入正式源码仓库。

## 1. 正式处理流程

```text
新建项目
  → 上传比赛视频 / 球员名单
  → 视频健康检查与参数配置
  → 多锚点动态标定
  → 人 / 球检测、追踪与技术 ID
  → 身份质量审计、队伍提示与多帧号码识别
  → 米制跑动、球权与传球候选
  → 人工复核与真实身份确认
  → 事件、高光、2D 回放与球员卡
  → 正式比赛报告与结果归档
```

产品界面保留四个老板侧进度阶段：**追踪中 → 号码识别 → 事件检测 → 报告生成**。内部模块可以更细，但不得维护第二套生产主管线。

## 2. 快速部署

Windows 新电脑：双击 `DEPLOY_ONE_CLICK_WINDOWS.bat`，等到浏览器打开 `http://127.0.0.1:8000`。数据写在 `runtime/projects/`。

Docker 场馆部署：`DEPLOY_DOCKER_WINDOWS.bat` 或 `./deploy.sh`。完整说明见 [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)。

## 3. 仓库结构

```text
app/features/              按项目、任务、身份、分析、产物划分的产品功能
app/services/              旧接口兼容层与编排服务
engine/
  tracking/                检测、追踪、场内过滤、技术 ID、候选事件
  football_metric_running/ 动态 Homography、米制坐标、跑动与速度
  match_analysis/          OCR、球权、传球、事件、球员卡与报告数据
  identity_audit/          技术 ID 外观污染/切换候选审计
  identity_resolution/     动态标定门控、BF16 全参 ReID、两阶段合并与隔离
  repository_qa/           仓库级验证入口
scripts/                   安装、检查、发布与验收脚本
docs/                      用户、部署、架构、开发、运维、验收与发布文档
deploy/                    Docker、PostgreSQL 分区、Redis 与 MinIO 部署
models/                    检测模型缓存；容器启动时自动准备
runtime/                   仅开发兼容；正式部署使用数据库与对象存储
examples/                  不含真实比赛/球员数据的配置或调用示例
```

`demo_data/` 不属于正式源码交付内容，并由仓库规则明确禁止重新提交。

## 4. 数据与模型边界

以下内容禁止进入 Git/Gitee：

- 完整或裁剪比赛视频、球员素材与真实名单数据；
- MOT/追踪结果、事件索引、切片、热力图、报告等运行产物；
- `.pt/.pth/.onnx/.engine` 等模型权重；
- `.env`、Token、密钥、账号口令；
- Demo/参考比赛数据；
- 单文件超过 10 MB 的交付外大文件。

需要共享的大文件应放公司批准的文件存储，并在交付说明中登记位置与校验值。

## 5. 开发与验证

基础验证：

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

完整 AI 推理需要额外安装 `requirements-ai.txt` 与模型权重。仓库级测试通过不等价于目标 Windows GPU 上的新比赛已经完成全链路验收；正式发布仍需按 [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) 执行真实新素材验证。

## 6. 文档导航

- [`docs/README.md`](docs/README.md)：文档索引
- [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)：产品使用流程
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)：部署与环境准备
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)：架构、数据流与模块职责
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)：开发规范与测试要求
- [`docs/仓库使用规范.md`](docs/仓库使用规范.md)：提交纪律、分支与数据红线
- [`docs/调参清单.md`](docs/调参清单.md)：生产参数登记表
- [`docs/PIPELINE_BASELINE.md`](docs/PIPELINE_BASELINE.md)：老板最初 POC 管线与当前生产实现映射
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md)：运行、备份、恢复与排错
- [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md)：正式验收
- [`docs/RELEASE.md`](docs/RELEASE.md)：发布与 Gitee 镜像检查项

## 7. 维护原则

`app/services/pipeline.py` 是产品层唯一总编排入口；各引擎 CLI 只用于被编排、诊断或测试。参数变更必须同步更新 `docs/调参清单.md`，关键源码变更后必须重新生成并验证 `CHAIN_AUDIT.json`。
