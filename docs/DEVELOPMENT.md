# 开发指南

## 命名原则

目录、包、脚本和运行产物按**职责/数据类型**命名，不按周次、日期、冲刺阶段命名。示例：

- `tracking_lib`，不要使用阶段号包名；
- `run_integrated_analysis.py`，不要把迭代编号写进入口；
- `demo_data/reference_match`，不要用某次演示日期作为长期目录；
- `tracking_run_metadata.json`，不要把研发阶段号写入数据契约。

版本号可以存在于发布 metadata 中，但稳定代码路径不应依赖版本或排期名称。

## 公共代码原则

1. 同一实现只保留一个 canonical source。
2. 跨模块共享时使用适配器/明确依赖，不复制源文件。
3. vendor/third-party 代码不与 first-party 去重混在一起；许可证和上游边界必须保留。
4. 删除近似重复代码前先比较行为和测试，不以文件名或行数相似度直接判定。
5. PyTorch、Ultralytics 等重量级 AI 依赖放在真正执行 AI 流程的运行边界加载；纯数据转换、导出和校验工具不应仅因导入同一 CLI 就被迫安装完整 AI 环境。

## 本地测试

```bash
python -m compileall -q app engine scripts
python -m pytest -q app/tests
(cd engine/tracking && python -m pytest -q)
(cd engine/football_metric_running && PYTHONPATH=src python -m pytest -q)
(cd engine/identity_audit && PYTHONPATH=../tracking:. python -m pytest -q)
(cd engine/match_analysis && PYTHONPATH=. python -m pytest -q)
```

米制模块的完整测试会导入 Ultralytics；若开发机没有 AI 依赖，至少运行不依赖检测器的测试并在 CI/目标环境补齐完整套件。

## 发布前检查

```bash
python scripts/validate_windows_launchers.py
python scripts/audit_engine_chain.py
python scripts/verify_product.py
python scripts/acceptance_check.py
python scripts/generate_release_metadata.py
```

`CHAIN_AUDIT.json` 是重构后 first-party 关键文件的完整性清单。修改关键源码后必须重新生成，不能为了让检查变绿而手工改某一条哈希。

## 新增功能放哪里

- Web/API/状态持久化：`app/`
- 检测/追踪/场内过滤：`engine/tracking/`
- 标定/米制运动：`engine/football_metric_running/`
- OCR/球权/传球/球员卡/报告数据：`engine/match_analysis/`
- 身份质量审计：`engine/identity_audit/`
- 安装、检查、发布工具：`scripts/`

如果功能跨两个引擎，先确认是否应该提取小型共享接口；不要建立一个带历史阶段名的新复制目录。
