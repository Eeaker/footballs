# PROTOTYPE：自动球衣颜色模式

问题：在不人工指定球衣颜色的前提下，自动发现实际颜色簇，是否比强制 K=2 更适合当前低机位 U12 视频？

运行：

```powershell
python prototypes/team_color_modes/run.py --video <比赛.mp4> --mot <tracking_mot.txt> `
  --evaluation <clip_eligibility.json> --output <必须不存在的新目录>
```

这是一次性实验，不会写入正式 `player_team_map.csv`。

