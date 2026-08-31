# Match Analysis Engine

负责追踪后的正式赛后分析：号码 OCR、队伍映射、米制球权、球权转换、传球候选、球员卡数据与报告输入。

## 正式入口

产品层调用：

```bash
python run_integrated_analysis.py \
  --tracking-dir <tracking-output> \
  --calibration <validated-dynamic-calibration.json> \
  --video <source-video> \
  --running-src ../football_metric_running/src \
  --output <new-output-dir>
```

该入口只生成一次正式数据结果：

- `analysis/`：球权、球权转换、传球及质检数据；
- `metric_running/`：逐帧米制坐标与跑动统计；
- `integrated_manifest.json`：关键输入、结果与 SHA-256 清单。

2D 球场回放不在此阶段重复渲染；正式产品在报告阶段由 `render_metric_pitch.py` 统一生成 `metric_pitch_replay.mp4`。

号码识别入口：

```bash
python run_jersey_ocr.py --video <video> --mot <tracking_mot.txt> --team-hints <team_hints.csv> --output <new-output-dir>
```

## 传球口径

1. 球与球员脚点在同一帧投影到米制球场；
2. 距离和连续帧门槛形成稳定球权；
3. A→B 的所有稳定球权变化记录为 transition；
4. 只有同队、稳定且达到米制位移阈值的主动定向候选进入 pass network；
5. 传球保留人工复核，不把代理条件包装成绝对真值。

## 共享代码

`analysis_lib/tracking_adapter.py` 复用 `tracking/tracking_lib` 的 actor / homography / team-feature 实现，不保存复制版追踪源码。

第三方来源和许可证见 `THIRD_PARTY_NOTICES.md` 与 `licenses/`。
