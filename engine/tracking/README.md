# Tracking Engine

负责人物/足球检测与追踪、场内过滤、技术 ID、球位置和候选事件。

## 新视频适配

```powershell
python onboard.py --video <video> --venue <venue> --weights <yolov8x.pt> --expected-players <n> --team-clusters 3 --device 0
```

确认健康检查、球场区域、队色样本和短片试跑后：

```powershell
python run_pipeline.py --config <onboard-config.yaml> --output <new-output-dir> --device 0
```

底层追踪入口为 `run_tracking.py`，公共实现位于 `tracking_lib/`。

## 关键产物

- `tracking_mot.txt`：技术 ID 轨迹；
- `tracking_vis.mp4`：追踪可视化；
- `ball_positions_observed.csv`：球位置；
- `tracking_run_metadata.json`：帧域/FPS/本次运行元数据；
- `event_index.json` / `event_index.csv`：候选事件索引。

技术 ID 只在一次运行内有意义，不应直接当成真实球员身份。
