# 场馆部署指南

## 安装（推荐：联网一键）

新 Windows 电脑只双击根目录 **`DEPLOY_ONE_CLICK_WINDOWS.bat`**。不要先装 Docker 或 Python，不要再运行第二个安装脚本。

该入口会连续完成：从零检查本机（不使用系统 Python/pip）→ 下载官方 3.12.10 到 `runtime/python`（不改系统 PATH，缺少 pip 时 `ensurepip` 自举）→ 创建 `.venv` → 安装 VC++ x64 运行库、Web/AI 依赖和默认 `yolov8x.pt` → 有 NVIDIA 则装 GPU 版 PyTorch，失败则自动回退 CPU → 创建桌面快捷方式 → 启动并打开 `http://127.0.0.1:8000`。

中间没有模式选择，也不要再运行第二个安装脚本。只有失败才停在窗口里。第一次必须联网（Python 约 26 MB，PyTorch/CUDA 与模型合计数 GB）。之后再点同一个文件只启动服务。数据写在 `runtime/projects/`，不依赖 PostgreSQL。GPU 分析仍需本机 NVIDIA 驱动。需要重装运行时时用 `REPAIR_WINDOWS.bat`。

完全离线的第二台电脑：先在联网 Windows 上运行 `PREPARE_OFFLINE_WINDOWS.bat`，再拷贝整个目录，在离线机双击 `INSTALL_OFFLINE_WINDOWS.bat`。

## Docker 场馆部署（可选）

需要 PostgreSQL / Redis / MinIO 时：Windows 双击 `DEPLOY_DOCKER_WINDOWS.bat`；Linux / macOS 运行 `./deploy.sh`。若本机还没有 Docker，Windows 会尝试用 winget 安装 Docker Desktop。正式 BF16 分析还要求最新版 NVIDIA 驱动和 NVIDIA Container Toolkit。macOS 可运行管理与结果浏览，正式分析应调度到 NVIDIA 节点。

## 数据职责

| 数据 | 持久化位置 | 说明 |
|---|---|---|
| 项目、任务状态 | PostgreSQL `core` | JSONB 元数据，8 路哈希分区 |
| 每帧轨迹 | PostgreSQL `tracking` | 按比赛分区 |
| 事件和结果 | PostgreSQL `analytics` | 按比赛分区 |
| 合并、隔离、球员 | PostgreSQL `identity` | 未合并 ID 不进入球员表 |
| 视频、标定、图片、报告 | MinIO | S3 对象，不依赖应用本地磁盘 |
| 并发协调 | Redis | 分布式任务租约，不保存业务事实 |

`/workspace` 是容器内临时处理区，结果在任务结束后写入 PostgreSQL 与 MinIO，容器重启时可从对象存储恢复。

## 身份发布规则

- `expected_players` 只是先验，不按人数强制合并。
- 混人框、错误目标框和错误脚点隔离，不参与训练、ReID 投票或运动关联。
- 短碎片保留观测；证据不足时作为隔离 ID，不生成球员。
- 只有至少两个碎片 ID 构成的已确认合并组进入球员中心。
- 显示名采用 OCR 确认后的队伍和号码，例如“蓝队25号”。
- 头像从真实轨迹按清晰度、曝光、像素尺寸和遮挡率选取。

## 查看状态

```sh
docker compose --env-file deploy/.env -f deploy/compose.yaml ps
docker compose --env-file deploy/.env -f deploy/compose.yaml logs -f app
```

升级前备份 `postgres-data`、`redis-data` 和 `object-data` 三个命名卷。
