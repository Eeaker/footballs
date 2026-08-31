# 足球视频分析与米制跑动 Demo

本仓库整合两条已经跑通的流水线：

1. 球员/足球检测、跟踪、全局 ID、事件索引与事件切片。
2. 固定光心左右旋转机位下的动态单应标定和球员米制跑动统计。

当前版本是可复现的工程 Demo，不是职业比赛 GPS 级体能系统。参考四点标定的
独立长度误差为 0.890 m，中心圈半径暂按 3 m，正式应用前仍需补充真实尺寸、
多位置 0.5 m 标定验收、人工轨迹真值和 tracking ID 到真实球员的合并。

## 已完成结果

- 视频：1920×1080、30 FPS、62,204 帧、34:33.47。
- 统一帧域：检测、跟踪、事件和切片统一使用零起始 `proc_idx`。
- 动态标定：12,441 个姿态样本全部接受，62,204 帧均有动态 H。
- 跑动口径：脚点 → 米制 H → 11 帧中值 → 总距离/P95 速度/高速跑距离。
- 测试：帧域与米制模块合计 16 项通过。

## 目录

```text
src/
├── pipeline.py                       # 检测、跟踪、ReID、事件与切片主流水线
└── running_metrics_v1/               # 标定、动态 H、米制统计与可视化
tests/                                # 帧域和跑动指标测试
artifacts/
├── pipeline/                         # MOT、事件索引、对比表和代表切片
└── metric_running/                   # 标定、动态 H、CSV、质量报告与 QA
large_artifacts/                      # Git LFS 管理的 720p 全长检查视频
scripts/verify_release.ps1            # 提交前验收
ARTIFACTS.md                          # 文件范围、SHA-256 和未入库大文件
```

## 环境

推荐 Python 3.9–3.11 和 NVIDIA GPU：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .
```

如需 GPU 版 PyTorch，请按本机 CUDA 版本先安装官方对应 wheel。

## 运行主流水线

原始视频和 YOLO 权重未纳入仓库，文件哈希见 `ARTIFACTS.md`：

```powershell
python src/pipeline.py `
  --video path\to\match.mp4 `
  --weights path\to\yolov8x.pt `
  --outdir outputs `
  --device 0 `
  --vid_stride 1
```

完整参数可通过 `python src/pipeline.py --help` 查看。

## 使用仓库内产物复算米制跑动

```powershell
python -m running_metrics_v1.calculate_running `
  --mot artifacts/pipeline/tracking_mot.txt `
  --calibration artifacts/metric_running/full_run/dynamic_calibration_45x25.json `
  --outdir outputs/metric_running
```

注意：动态标定 JSON 中仍记录原开发机的视频绝对路径。重新渲染视频前，应把其中
`video` 字段改为本机原视频路径；仅复算 CSV 不读取视频。

生成短检查片段：

```powershell
python -m running_metrics_v1.render_demo `
  --calibration artifacts/metric_running/full_run/dynamic_calibration_45x25.json `
  --timeseries artifacts/metric_running/full_run/metrics/player_running_timeseries.csv `
  --start-proc 54000 --end-proc 54090 `
  --output outputs/check_1800s.mp4
```

## 测试

```powershell
python -m pytest -q
powershell -ExecutionPolicy Bypass -File scripts/verify_release.ps1
```

## GitHub 与大文件

普通 Git 会阻止超过 100 MiB 的文件。本仓库通过 `.gitattributes` 将 MP4 交给
Git LFS；两个入库的全长视频均小于 GitHub Free/Pro 当前单文件 2 GB 上限。

```powershell
git lfs install
git lfs track "*.mp4"
git add .gitattributes .
git status
```

不要通过 GitHub 网页直接上传本目录；网页上传不会替你正确处理 LFS。1080p 米制
成片和原 1080p 跟踪成片均超过 2 GB，未复制到本仓库，建议放 GitHub Release
之外的对象存储或网盘，并使用 `ARTIFACTS.md` 中的 SHA-256 校验。

GitHub 官方说明：

- https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage

## 第三方说明

第三方投影约定和许可信息见 `THIRD_PARTY_NOTICES.md`。

## 许可提醒

仓库尚未选择项目级开源许可证。公开前需要由项目负责人决定许可证，并确认原视频、
球员画面、模型权重和训练/测试数据是否具有公开授权。未选择许可证不等于允许他人
自由复制、修改或再分发。
