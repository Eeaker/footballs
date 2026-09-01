# 文档索引

正式仓库文档按“使用 → 部署 → 架构 → 开发 → 运维 → 验收 → 发布”组织。

| 文档 | 适用对象 | 内容 |
|---|---|---|
| [`USER_GUIDE.md`](USER_GUIDE.md) | 使用人员 | 建项目、上传、标定、分析、复核、结果与导出 |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | 部署/运维 | Windows、GPU、Python、模型、离线安装、故障排查 |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | 开发/技术负责人 | 模块职责、生产数据流、关键数据契约与恢复策略 |
| [`DEVELOPMENT.md`](DEVELOPMENT.md) | 开发人员 | 分支、代码组织、测试、发布前检查 |
| [`仓库使用规范.md`](仓库使用规范.md) | 全员 | 提交纪律、分支规则、数据/模型/密钥红线 |
| [`调参清单.md`](调参清单.md) | 算法/产品负责人 | 生产参数、影响范围与变更登记要求 |
| [`PIPELINE_BASELINE.md`](PIPELINE_BASELINE.md) | 技术负责人 | 最初老板 POC 管线与当前 canonical 实现映射 |
| [`OPERATIONS.md`](OPERATIONS.md) | 运维/现场人员 | 日常运行、备份、恢复与排错 |
| [`ACCEPTANCE.md`](ACCEPTANCE.md) | 验收人员 | 仓库级与真实新素材验收边界 |
| [`RELEASE.md`](RELEASE.md) | 发布人员 | 正式源码快照、Gitee 镜像和发布门禁 |
| [`MIGRATION.md`](MIGRATION.md) | 维护人员 | 历史结构迁移说明，仅用于追溯 |

## 文档维护要求

- 参数默认值变化：同步更新 `调参清单.md`。
- 目录或数据契约变化：同步更新 `ARCHITECTURE.md`、README 和相关用户/开发文档。
- 部署方式变化：同步更新 `DEPLOYMENT.md` 与启动脚本说明。
- 验收口径变化：同步更新 `ACCEPTANCE.md`。
- 发布边界变化：同步更新 `RELEASE.md` 与 `.gitignore`。

历史迁移说明不应作为新人或正式用户的首要入口；正式使用以 README 与上述当前文档为准。
