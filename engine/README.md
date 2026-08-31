# Analysis Engine

`engine/` 只包含算法/质量模块，产品编排入口在 `app/services/pipeline.py`。

| 目录 | 责任 |
|---|---|
| `tracking/` | 检测、追踪、场内过滤、技术 ID、候选事件 |
| `football_metric_running/` | 动态标定、米制坐标、跑动/速度 |
| `match_analysis/` | OCR、球权/传球、2D 球场、球员卡、报告数据 |
| `identity_audit/` | 技术 ID 外观质量审计，只输出候选 |
| `repository_qa/` | 跨模块验证 |

共享 first-party 实现只保留 canonical source；`match_analysis` 通过适配器复用 tracking 公共能力，不复制源码。
