# 运维与恢复

## 运行数据

所有正式项目位于 `runtime/projects/<project_id>/`。`runtime/` 不进入版本控制，升级代码时不要用“删除整个项目目录”的方式修复环境。

## 备份

优先备份整个项目目录；至少保留：

- `input/` 原视频与名单；
- `calibration/` 动态标定；
- `config/` 本次参数；
- `reviews/` 人工确认；
- `outputs/` 与 `exports/`。

## 中断与重试

服务重启后，运行中的任务会被标记为中断。用户可从失败阶段继续；恢复逻辑会校验前置阶段关键文件，并清理失败阶段的残留输出。不要手工把不完整文件改名成“完成”产物。

## 常见故障定位

- **服务起不来**：先运行 `CHECK_WINDOWS.bat`。
- **Torch/CUDA 错误**：运行 `DIAGNOSE_WINDOWS.bat`，必要时 `REPAIR_WINDOWS.bat`。
- **模型缺失**：运行 `DOWNLOAD_MODEL_WINDOWS.bat` 或在系统状态页上传。
- **米制分析不启动**：检查动态标定的视频元数据、尺度验证和覆盖率。
- **阶段恢复失败**：检查对应项目 `logs/` 和 `outputs/` 中被报告缺失的关键产物。

## 归档

正式交付使用项目导出 ZIP。导出包包含结构化数据、报告和 artifact manifest；模型、系统 Python、CUDA 与虚拟环境不属于项目归档。
