# Match Analysis Engine

负责追踪后的赛后分析：队伍提示、号码 OCR、米制球权、球权转换、传球候选、2D 球场视频、球员卡和报告数据。

## 集成入口

```bash
python run_integrated_analysis.py \
  --tracking-dir <tracking-output> \
  --calibration <validated-dynamic-calibration.json> \
  --video <source-video> \
  --running-src ../football_metric_running/src \
  --output <new-output-dir>
```

输出包含 `analysis/`、`metric_running/`、`integrated_manifest.json` 和 2D 球场视频。

基础球权/传球入口：

```bash
python run_analysis.py --tracking-dir <tracking-output> --calibration <calibration> --video <video> --output <new-output-dir>
```

号码识别入口：

```bash
python run_jersey_ocr.py --video <video> --mot <tracking_mot.txt> --team-hints <team_hints.csv> --output <new-output-dir>
```

## 传球口径

1. 球与球员脚点在同一帧投影到米制球场；
2. 距离和连续帧门槛形成稳定球权；
3. A→B 的所有稳定球权变化记录为 transition；
4. 只有同队、稳定且达到米制位移阈值的主动定向候选进入 pass network；
5. 传球仍保留人工抽检，不把代理条件包装成绝对真值。

## 共享代码

`analysis_lib/tracking_adapter.py` 复用 `tracking/tracking_lib` 的 actor / homography / team-feature。这里不再保存复制版追踪源码。

第三方来源和许可证见 `THIRD_PARTY_NOTICES.md` 与 `licenses/`。
