# 部署指南

## 目标环境

- Windows 10/11 x64
- 64 位 Python 3.11 或 3.12
- 推荐 NVIDIA GPU
- 正式 AI 推理需要 `yolov8x.pt` 与 AI 依赖；Web/结果浏览可在 CPU 环境运行

## 推荐安装

首次直接运行 `RUN_WINDOWS.bat`。它会检查 `.venv` 和安装完成标记；环境未完成时调用 `INSTALL_WINDOWS.bat`，成功后启动系统。

也可以分开执行：

```text
INSTALL_WINDOWS.bat
START_WINDOWS.bat
```

安装脚本会创建项目私有 `.venv`，自动检测 NVIDIA 环境并选择 PyTorch 安装方案。若已有健康 PyTorch，会优先复用，避免无意义的大文件重复下载。

## 模型

检测权重默认位于：

```text
models/yolov8x.pt
```

模型不进入 Git 仓库。可运行 `DOWNLOAD_MODEL_WINDOWS.bat`，也可从“系统状态”页面上传。缺少模型时产品层仍可启动，但新视频正式推理的 preflight 不会通过。

## 离线电脑

在联网 Windows 机器运行 `PREPARE_OFFLINE_WINDOWS.bat` 生成 `wheelhouse/`，再把完整目录复制到离线目标机，运行 `INSTALL_OFFLINE_WINDOWS.bat`。目标机仍需提前具备兼容的 Python 与 NVIDIA 驱动。

## 静默演示启动

`PRESENT_WINDOWS.vbs` 使用 `pythonw.exe` 在后台启动服务并打开浏览器；`STOP_WINDOWS.bat` 根据 `runtime/server.pid` 停止后台服务。该 VBS 文件现在显式纳入版本控制，不再被 `*.vbs` 规则误排除。

## PyTorch / c10.dll 排错

Windows 出现 `WinError 1114`、`torch\lib\c10.dll` 或 CUDA 初始化失败时：

1. 运行 `DIAGNOSE_WINDOWS.bat`；
2. 检查 64 位 Python 3.11/3.12；
3. 检查 Microsoft Visual C++ v14 x64 Runtime；
4. 运行 `REPAIR_WINDOWS.bat` 让安装器重新验证/修复 PyTorch；
5. 查看 `runtime/diagnostics/windows_torch_probe.json`。

安装器不仅检查 `torch.cuda.is_available()`，还会执行 GPU 矩阵运算以验证实际 kernel。

## 启动前检查

```text
CHECK_WINDOWS.bat
```

检查包括启动器编码、Python/依赖、first-party 代码完整性和产品 API。GPU/模型是正式新素材推理条件，不应与“Web 包能启动”混为一个结论。
