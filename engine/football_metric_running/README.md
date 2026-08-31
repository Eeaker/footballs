# Metric Running Engine

该模块只负责正式产品中的**动态球场标定与米制跑动计算**。检测、追踪、ReID 和事件检测由 `engine/tracking` 与 `engine/match_analysis` 负责，不在这里维护第二套生产流水线。

## 正式职责

- 根据人工锚点构建动态 Homography；
- 将追踪脚点从像素坐标投影到球场米制坐标；
- 对轨迹进行稳健平滑；
- 计算总距离、速度分布和高速跑距离；
- 输出逐帧米制时序和球员汇总。

## 正式调用

产品层通过 `engine/match_analysis/run_integrated_analysis.py` 调用：

```bash
python -m running_metrics_v1.calculate_running \
  --mot <tracking_mot.txt> \
  --calibration <dynamic_calibration.json> \
  --outdir <metric-running-output>
```

多锚点动态标定由：

```bash
python -m running_metrics_v1.build_multi_anchor_dynamic_calibration ...
```

生成。

## 核心源码

```text
src/running_metrics_v1/
├─ calculate_running.py
├─ metrics.py
├─ mot.py
├─ homography.py
├─ dynamic_calibration.py
├─ build_multi_anchor_dynamic_calibration.py
├─ build_rotation_dynamic_calibration.py
├─ calibrate_pitch.py
├─ evaluate_rotation_registration.py
└─ render_demo.py
```

`src/pipeline.py` 这类曾同时承担检测、追踪、事件和米制计算的独立历史流水线已经移除，避免与正式 Tracking / Match Analysis 重复。

## 质量边界

米制结果必须建立在通过验证的动态标定之上。标定覆盖率或尺度验证不通过时，正式产品应停止米制链路，而不是继续输出伪精确距离。
