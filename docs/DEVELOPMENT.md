# 开发指南

## 1. 分支与提交

- 稳定主线：`main`。
- 日常开发：`dev-姓名缩写` 或 `feature/*`。
- 发布整理：`release/*`。
- Commit message：`[模块] 改动内容 — 原因`。
- 合并前必须附自测命令和结果；关键逻辑变化需要对应测试或验收记录。

完整仓库纪律见 [`仓库使用规范.md`](仓库使用规范.md)。

## 2. 代码组织原则

目录、包、脚本和运行产物按**职责/数据类型**命名，不按周次、日期、冲刺阶段命名。

1. 同一实现只保留一个 canonical source。
2. 跨模块共享时使用适配器或明确依赖，不复制源文件。
3. `app/services/pipeline.py` 是产品层唯一总编排入口；模块 CLI 用于被编排、诊断或测试。
4. vendor/third-party 代码与 first-party 代码边界必须明确，并保留许可证信息。
5. 删除近似重复代码前先比较行为和测试，不以文件名或行数相似度直接判定。
6. PyTorch、Ultralytics 等重量级依赖放在真正执行 AI 流程的运行边界加载；纯数据转换、导出和校验工具不应被迫依赖完整 AI 环境。

## 3. 新增功能放置

- Web/API/状态持久化：`app/`
- 检测/追踪/场内过滤：`engine/tracking/`
- 标定/米制运动：`engine/football_metric_running/`
- OCR/球权/传球/事件/球员卡/报告数据：`engine/match_analysis/`
- 身份质量审计：`engine/identity_audit/`
- 安装、检查、发布工具：`scripts/`
- 产品、部署、架构与验收文档：`docs/`

功能跨多个引擎时，优先提取小型共享接口，不建立带历史阶段名的新复制目录。

## 4. 参数与数据契约

参数变更必须同步维护 [`调参清单.md`](调参清单.md)。稳定数据契约（例如 MOT、`tracking_run_metadata.json`、`event_index.json`）发生字段或语义变化时，需要同时更新：

- 生产代码；
- 读取/导出端；
- 测试；
- 架构/用户文档；
- 如属于关键源码，重新生成 `CHAIN_AUDIT.json`。

## 5. 本地测试

```bash
python -m compileall -q app engine scripts
python -m pytest -q app/tests
(cd engine/tracking && python -m pytest -q)
(cd engine/football_metric_running && PYTHONPATH=src python -m pytest -q)
(cd engine/identity_audit && PYTHONPATH=../tracking:. python -m pytest -q)
(cd engine/match_analysis && PYTHONPATH=. python -m pytest -q)
```

米制模块的完整测试会导入 AI 依赖；如果开发机没有对应环境，至少执行可运行的轻量测试，并在 CI/目标 Windows GPU 环境补齐完整套件。

## 6. 发布前检查

```bash
python scripts/validate_windows_launchers.py
python scripts/audit_engine_chain.py
python scripts/verify_product.py
python scripts/acceptance_check.py
python scripts/generate_release_metadata.py
```

此外必须执行发布快照检查：

- 无 `demo_data/`；
- 无比赛视频/球员数据/运行产物；
- 无模型与密钥；
- 无单文件 >10 MB；
- README、部署、参数与验收文档与实际代码一致。

`CHAIN_AUDIT.json` 是 first-party 关键文件完整性清单。关键源码变化后应由工具重新生成/更新，不能为了让检查通过而手工伪造哈希。
