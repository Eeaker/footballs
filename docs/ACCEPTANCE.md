# 验收清单

## A. 仓库/产品包

- [ ] `python -m compileall -q app engine scripts` 通过。
- [ ] App、tracking、metric running、identity audit、match analysis 的对应测试通过。
- [ ] `python scripts/validate_windows_launchers.py` 通过。
- [ ] `python scripts/audit_engine_chain.py` 通过。
- [ ] `python scripts/verify_product.py` 通过。
- [ ] 仓库不存在文件/目录名中的阶段号命名。
- [ ] 不存在 `football_metric_running` 的同源副本。
- [ ] `PRESENT_WINDOWS.vbs` 实际存在且为 ASCII。

## B. 产品功能

- [ ] 可创建项目并上传视频/名单。
- [ ] 可保存、导入、导出分析参数。
- [ ] 可上传或创建多锚点动态标定，未过门控时禁止米制分析。
- [ ] 四阶段进度、失败恢复和运行日志正常。
- [ ] 结果中心包含总览、2D 回放、事件、高光、球员、质量/复核、报告和导出。
- [ ] 可人工确认传球、技术 ID、队伍语义和八维评分。
- [ ] 可导出结果 ZIP。

## C. 目标 Windows GPU 新素材

仓库测试与 demo 验收**不能**替代真实新素材推理。目标机器还应验证：

- [ ] `models/yolov8x.pt` 有效；
- [ ] CUDA/PyTorch 实际 GPU 运算通过；
- [ ] 一段从未预生成结果的新视频完整跑完；
- [ ] 运行 `python scripts/verify_fresh_project.py <project_id>` 通过；
- [ ] 人工抽查追踪、号码、传球和跑动口径；
- [ ] 最终报告和 ZIP 可打开，artifact manifest 完整。

`verify_fresh_project.py` 证明的是“这次项目确实完整跑过”，不代表算法精度自动达标。
