# 足球 ReID 两阶段流程

后续开发统一在本目录。新运行默认写到同级 reid_pipeline_runs/时间戳，输入与原模型不被覆盖。

2026-09-10：用户要求当前只完成适配，不重新训练。默认质量配置已绑定审核过的输入指纹，排除ID114对应的混人观测；初始合并入口也在事件统计前执行该排除。人工身份确认与训练裁剪放行分开保存。

`scripts/prepare_reviewed_hard.py` 按逐帧白名单准备训练窗内的人工审核困难样本，验证/测试帧拒绝进入。`scripts/smoke_hard_augmentation.py` 默认仅生成两套增强配置，训练步数为0；只有后续明确授权并显式使用 `--smoke` 或 `--run-comparison` 才执行参数更新。新增强保持可选，当前正式权重不变。

## 运行

```powershell
python -B run_pipeline.py --device cuda
python -B run_pipeline.py --device cuda --strict-association
python -B run_pipeline.py --device cuda --stage 2
python -B scripts/train_selftrain.py --device cuda --prepare-only
python -B scripts/train_selftrain.py --device cuda --smoke
python -B scripts/train_selftrain.py --device cuda --rounds 1
```

默认执行优化二阶段关联并渲染，不训练。--strict-association使用旧保守关联器；--second-stage显式选择当前二阶段。stage 1/4只做提特征和关联，2只提特征，3只自训练，5只渲染且必须指定--tracking。--output指定新的外部运行目录。
入口从任意工作目录都可使用。需要PyTorch、torchvision、numpy、OpenCV、Pillow、timm、yacs；使用已有后端reid_model/vendor/TransReID。本机football-main项目的.venv环境已实际验证。

## 数据与模型

第一阶段输入是data/audit_data.pkl中已经完成的纯净轨迹合并，不在第二阶段擅自重写。内部视频零基0–1799对应原视频3150–4949；MOT为一基1–1800，当前审核数据30 FPS。
weights/reid_best.pth为原全参微调第21轮最佳模型；当前优化没有重新训练。实际编码是3840维TransReID JPM，不能把audit里旧512维缓存当作训练后的模型。缓存绑定视频、audit、权重、配置和时间窗，来源变化必须重新提取。
原训练、验证、测试及模型结构均保持不变。模型适用范围是本场已知的9个身份，不代表跨比赛能力。

## 二阶段与推理采样

配置集中configs/pipeline_config.json，模型预处理在configs/reid_config.json。

- quality：严格训练采样；端点默认去除3帧、重叠上限15%，自训练仍用此配置。
- inference_quality：推理端点不固定剔除，短于45帧的节点尽量补足4张干净观测；依然保留置信度、尺寸、重叠与画面边界检查。推理裁剪不自动变成训练样本。
- second_stage：短间隔比较相邻端点附近多帧，长间隔做身份找回，最长候选间隔1800帧；所有合并仍检查实际同帧、球衣、观测可达性、整组一致性与互斥候选歧义。
- use_training_references：默认使用原训练时段108张注册图生成9个参考中心。只从原训练分区提取，验证manifest/audit摘要并绑定模型权重；不读取待关联ID的人工对应表。模型变化后重新生成。当前参考准入为相似度≥0.70、第一与第二名差≥0.25、多帧投票一致率≥60%，同时保留其他合并约束。这些是当前开发视频上的实验参数，不能当成概率或跨视频保证。
- sparse_bridge：缺乏外观的短片段仅在同一个固定身份分量提供前后双端位置证据、可达且唯一时形成待审核连接；新续接片段不能反过来成为锚点继续扩张。

有注册图库时，报告会明确training_gallery_used=true、labels_used_for_association=true：这里使用的是原监督训练标签生成的图库。query_identity_map_used=false表示没有使用待关联ID的人工答案。不能把该模式宣传成完全无监督。
没有强制9分类或按9个输出停止的逻辑。隔离检测保留为灰色OCC，普通未知片段保留原身份。视频“ID X <- Y”表示新ID X来自原ID Y；橙色REVIEW表示待审核。
所有二阶段输出automatic_training_labels=false；不会因为合并成功就自动回灌自训练。

## 自训练

训练窗0–1049，验证1110–1379，测试1440–1799；先限制时间窗再提特征、建图、选伪标签。原训练图库及验证集固定。
只采用训练身份锚点和教师外观同时确认的新增样本，拒绝未知或冲突身份。每个身份均衡采样，每批至少一半原审核样本；保留随机遮挡、轻微旋转/仿射、亮度对比度和随机高斯模糊。验证不使用随机增强。
只有验证指标改善且安全指标不退化才晋升候选，最多两轮。--smoke做一次真实更新但禁止晋升。不要在同一批几乎相同的图片上无界重复训练。
当前这轮无需重跑原训练；如果继续改善ID181及门将等困难外观，先补同人其他时段难例与同队负例，并另留独立验收片段。若把原验证帧用于新训练，必须重新划分评测，不能继续沿用原验证成绩证明提升。

## 对照与结果

```powershell
python -B scripts/compare_association.py --features "旧运行/features/features.pkl" --enhanced-features "新运行/features/features.pkl" --reference-bank "新运行/features/reference_bank.npz" --output "新的外部实验目录"
```

程序先记录参数、执行全部方法，再读取人工对应表评测。不开--reference-bank就明确不使用注册图库。
本轮开发结果：已知ID分量20→14，同人配对恢复42/97→73/97，已知跨人误配0；全部普通分量33→25，保留13,111条检测。另有ID39、121、157三条未审核关联。完整9人尚未恢复，不能将开发集零错误宣传成工业级纯净保证。
26项回归测试、完整1800帧运行、模型摘要不变检查通过。最新报告在同级实验结论/二阶段完整优化_20260909。

## 项目边界

src/features.py负责采样、特征和训练图库；src/trajectory_reassociation.py保留严格基线；src/second_stage.py负责优化关联；src/selftraining.py负责训练闭环；src/render_video.py负责核对视频。scripts是入口，tests是回归检查。
data、weights与reid_model中的原训练数据/后端保留。旧output是历史结果，其中“输出9人”受人工映射与导出过滤影响，不作为算法评价。历史Linux环境和缓存已移到外部审计目录。
实验日志、视频、裁剪、临时文件和备份均放到代码目录外。
# 正式工作目录与初始合并（2026-09-09补充）

后续代码以本 `reid_pipeline` 为准。初始合并实现已迁入 `src/identity/`，不再运行或导入“最新纯净ID源码”中的代码。历史输入、旧版本和实验输出保留在外部。

- `scripts/refine_initial.py --audit <初始audit> --dense <dense证据> --output <新外部目录>`：包含隔离排除、歧义拒绝和补洞修复的初始合并。
- `scripts/compare_motion.py --features <特征缓存> --mapping <二阶段映射> --output <新外部目录>`：米制/图像坐标卡尔曼对照。
- `scripts/verify_refined_motion.py --audit <修复后的audit> --features <历史特征> --reference <冻结参考库> --output <新外部目录>`：验证节点及逐帧观测完全相同后复用特征，重跑二阶段和运动复核方案；节点变化则拒绝复用。

运动实验包含双端、分离主干联合检验、单侧复核三个层次。结果是 `requires_review` 候选，不自动成为训练标签。单侧方案的协方差尚未在独立时段标定，不能把相对最高分解释成身份正确概率。默认训练与主运行入口没有悄悄启用这些实验规则或替换旧数据。

人工标签绑定历史输出ID；修复初始合并可能改变ID组成。完整实验通过不变节点及观测溯源后才评估历史身份，禁止直接套用旧编号。查看外部实验目录 `卡尔曼双坐标实测_20260909` 的说明与核对视频。
