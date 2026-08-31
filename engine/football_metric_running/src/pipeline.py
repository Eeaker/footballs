# -*- coding: utf-8 -*-
"""
比赛视频球员检测跟踪与事件切片 POC (v4)
本次改动:
  1. Stage 2 新增两级出场过滤:
     - 合并前剔除极短 tracklet (降噪 + 大幅减少候选对);
     - 合并后按出场时长排序, 只保留达标且排名前 max_ids(默认 35) 的 ID,
       硬性保证最终 ID 数 <= max_ids, 并剔除背景中偶现的非球员;
  2. Stage 4 切片抽样改为「按事件类型分层」, 使各类动作数量均衡;
  3. 被剔除 ID 的检测框不再参与渲染 / MOT / 球员速度信号。
  4. 统一帧域: Stage 1~4 的分析数据全部使用处理帧编号 proc_idx；
     vid_stride 只用于原视频 I/O 与处理帧率换算，禁止对处理帧再次抽帧。
"""

import os
import re
import cv2
import json
import math
import argparse
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from ultralytics import YOLO


def safe_name(s):
    """清洗文件名中的非法字符(斜杠等), 避免被当成路径分隔符。"""
    return re.sub(r'[\\/:*?"<>|()（）\s]+', '_', str(s)).strip('_')


# ============================================================
#  统一帧域规则
# ============================================================
# 分析域: detections / ball_pos / tracklets / total_frames 的帧号和长度
#         一律是 Stage 1 产出的处理帧(proc)口径。
# I/O 域: 只有顺序读取原视频时，才按 vid_stride 映射 raw -> proc。
# 时间域: 秒 = proc_idx / processed_fps(raw_fps, vid_stride)。
def processed_fps(raw_fps, vid_stride):
    """Return processed-video FPS; vid_stride is applied exactly once."""
    return float(raw_fps) / max(int(vid_stride), 1)


def is_sampled_raw_frame(raw_idx, vid_stride):
    """Match Ultralytics LoadImagesAndVideos: retrieve raw frame V-1, 2V-1, ..."""
    stride = max(int(vid_stride), 1)
    return (int(raw_idx) + 1) % stride == 0


def raw_frame_index_for_proc(proc_idx, vid_stride):
    """Map a zero-based processed-frame index to its zero-based raw frame index."""
    stride = max(int(vid_stride), 1)
    return (int(proc_idx) + 1) * stride - 1


def resolve_reid_stride(raw_fps, vid_stride, reid_stride, reid_interval_sec):
    """Keep ReID sampling stable in real time when an interval is configured."""
    if reid_interval_sec is None or reid_interval_sec == 0:
        return max(int(reid_stride), 1)
    return max(1, round(float(reid_interval_sec) * processed_fps(raw_fps, vid_stride)))


def _validate_proc_frame_domain(total_frames, detections, ball_pos):
    """Fail early when a caller mixes raw-frame keys into processed-frame arrays."""
    if total_frames < 0:
        raise ValueError("total_frames 必须是非负的处理帧数量")
    indices = [int(f) for f, *_ in detections]
    indices.extend(int(f) for f in ball_pos)
    if not indices:
        return
    lo, hi = min(indices), max(indices)
    if lo < 0 or hi >= total_frames:
        raise ValueError(
            "检测到混合帧域: detections/ball_pos 必须使用处理帧编号，"
            f"有效范围 0..{max(total_frames - 1, -1)}，实际范围 {lo}..{hi}"
        )


# ============================================================
#  ReID 外观特征提取器
# ============================================================
class ReIDExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.model = nn.Sequential(*list(backbone.children())[:-1]).to(device).eval()
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 128), antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def __call__(self, crops):
        if len(crops) == 0:
            return np.zeros((0, 2048), dtype=np.float32)
        batch = []
        for c in crops:
            if c.size == 0:
                c = np.zeros((10, 10, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            batch.append(self.tf(rgb))
        batch = torch.stack(batch).to(self.device)
        feats = self.model(batch).squeeze(-1).squeeze(-1)
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float32)


def color_feature(crop):
    if crop.size == 0:
        return np.zeros(16 * 16, dtype=np.float32)
    h, w = crop.shape[:2]
    torso = crop[int(0.15 * h):int(0.55 * h), int(0.15 * w):int(0.85 * w)]
    if torso.size == 0:
        torso = crop
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


# ============================================================
#  Stage 1: 检测 + 短时跟踪 + 累积外观特征
# ============================================================
def stage1_detect_track(args, reid):
    print("[Stage 1] YOLOv8 + BoT-SORT 跟踪, 提取外观特征 ...")
    model = YOLO(args.weights)

    meta_cap = cv2.VideoCapture(args.video)
    raw_fps = meta_cap.get(cv2.CAP_PROP_FPS) or 30.0
    meta_cap.release()
    effective_reid_stride = resolve_reid_stride(
        raw_fps, args.vid_stride, args.reid_stride, args.reid_interval_sec
    )
    print(
        f"    ReID 采样: 每 {effective_reid_stride} 个处理帧一次 "
        f"(约 {effective_reid_stride / processed_fps(raw_fps, args.vid_stride):.3f} 秒)"
    )

    detections = []
    ball_pos = {}
    tracklets = defaultdict(lambda: {
        "frames": set(),
        "emb_sum": np.zeros(2048, dtype=np.float32),
        "emb_cnt": 0,
        "col_sum": np.zeros(256, dtype=np.float32),
        "col_cnt": 0,
        "first": None, "last": None,
    })

    results = model.track(
        source=args.video,
        classes=[0, 32],
        tracker="botsort.yaml",
        persist=True,
        stream=True,
        conf=args.conf,
        iou=0.5,
        vid_stride=args.vid_stride,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    frame_idx = -1
    for r in results:
        frame_idx += 1
        frame = r.orig_img
        if r.boxes is None or len(r.boxes) == 0:
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        ids = (r.boxes.id.cpu().numpy().astype(int)
               if r.boxes.id is not None else np.full(len(cls), -1))

        ball_mask = cls == 32
        if ball_mask.any():
            bi = np.argmax(np.where(ball_mask, conf, -1))
            bx = xyxy[bi]
            ball_pos[frame_idx] = ((bx[0] + bx[2]) / 2,
                                   (bx[1] + bx[3]) / 2, float(conf[bi]))

        crops, crop_ids = [], []
        do_reid = (frame_idx % effective_reid_stride == 0)
        for i in range(len(cls)):
            if cls[i] != 0 or ids[i] < 0:
                continue
            x1, y1, x2, y2 = xyxy[i]
            tid = int(ids[i])
            w, h = x2 - x1, y2 - y1
            detections.append((frame_idx, tid, float(x1), float(y1),
                               float(w), float(h), float(conf[i])))
            tk = tracklets[tid]
            tk["frames"].add(frame_idx)
            tk["first"] = frame_idx if tk["first"] is None else tk["first"]
            tk["last"] = frame_idx
            if do_reid:
                x1i, y1i = max(0, int(x1)), max(0, int(y1))
                crop = frame[y1i:int(y2), x1i:int(x2)]
                crops.append(crop)
                crop_ids.append(tid)

        if crops:
            embs = reid(crops)
            for k, tid in enumerate(crop_ids):
                tk = tracklets[tid]
                tk["emb_sum"] += embs[k]
                tk["emb_cnt"] += 1
                tk["col_sum"] += color_feature(crops[k])
                tk["col_cnt"] += 1

        if frame_idx % 1000 == 0:
            print(f"    处理到帧 {frame_idx}, 当前 tracklet 数 {len(tracklets)}")

    total_frames = frame_idx + 1
    print(f"[Stage 1] 完成。总帧数 {total_frames}, 原始 tracklet 数 {len(tracklets)}")
    return detections, ball_pos, tracklets, total_frames


# ============================================================
#  Stage 2: 全局重关联 + 出场时长过滤
# ============================================================
def _finalize_tracklet(tk):
    emb = (tk["emb_sum"] / tk["emb_cnt"]) if tk["emb_cnt"] > 0 else tk["emb_sum"]
    n = np.linalg.norm(emb)
    emb = emb / n if n > 1e-6 else emb
    col = (tk["col_sum"] / tk["col_cnt"]) if tk["col_cnt"] > 0 else tk["col_sum"]
    return emb, col


def stage2_global_reassoc(tracklets, total_frames, args):
    # total_frames 已由 Stage 1 按处理帧计数，不能再次除以 vid_stride。
    proc_total = max(1, total_frames)
    print("[Stage 2] 全局重关联 (贪心合并到 max_ids=%d, 出场过滤) ..." % args.max_ids)

    # --- (a) 合并前剔除极短 tracklet: 降噪 + 减少候选对 ---
    all_ids = list(tracklets.keys())
    ids = [t for t in all_ids
           if len(tracklets[t]["frames"]) >= args.min_track_frames]
    print(f"    预过滤: {len(all_ids)} -> {len(ids)} "
          f"(剔除出场 < {args.min_track_frames} 帧的碎片 tracklet)")

    feats = {t: _finalize_tracklet(tracklets[t]) for t in ids}

    # --- (b) 计算候选合并对 (时间互斥 + 颜色不冲突) ---
    pairs = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            ta, tb = ids[a], ids[b]
            # 没有实际采到外观/颜色特征时不参与相似度合并。
            if (tracklets[ta]["emb_cnt"] == 0 or tracklets[tb]["emb_cnt"] == 0
                    or tracklets[ta]["col_cnt"] == 0 or tracklets[tb]["col_cnt"] == 0):
                continue
            fa, fb = tracklets[ta]["frames"], tracklets[tb]["frames"]
            if len(fa & fb) > 1:
                continue
            ea, ca = feats[ta]
            eb, cb = feats[tb]
            emb_sim = float(np.dot(ea, eb))
            col_sim = float(cv2.compareHist(ca.reshape(16, 16),
                                            cb.reshape(16, 16),
                                            cv2.HISTCMP_CORREL))
            if col_sim < args.color_min:
                continue
            sim = args.wa * emb_sim + args.wc * col_sim
            pairs.append((sim, ta, tb))
    pairs.sort(reverse=True)
    print(f"    候选合并对: {len(pairs)}")

    # --- (c) 贪心合并 ---
    parent = {t: t for t in ids}
    root_frames = {t: set(tracklets[t]["frames"]) for t in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_clusters = len(ids)
    merged = 0
    for sim, a, b in pairs:
        if n_clusters <= args.max_ids or sim < args.merge_floor:
            break
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        # Never merge tracklets that coexist in even one frame; otherwise the
        # MOT export can contain two boxes for one global identity and frame.
        if root_frames[ra] & root_frames[rb]:
            continue
        if len(root_frames[ra]) < len(root_frames[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        root_frames[ra] |= root_frames[rb]
        merged += 1
        n_clusters -= 1

    # --- (d) 统计各簇出场时长 ---
    cluster_members = defaultdict(list)
    cluster_frames = defaultdict(set)
    for t in ids:
        r = find(t)
        cluster_members[r].append(t)
        cluster_frames[r] |= tracklets[t]["frames"]

    # --- (e) 出场过滤: 按时长排序, 达标 + 取前 max_ids ---
    stats = []
    for r, fr in cluster_frames.items():
        ratio = len(fr) / proc_total
        stats.append((ratio, len(fr), r))
    stats.sort(reverse=True)

    kept_roots = []
    for ratio, cnt, r in stats:
        if ratio < args.min_presence_ratio:
            continue
        if len(kept_roots) >= args.max_ids:
            break
        kept_roots.append(r)

    # 可选兜底，默认关闭，避免把低于 min_presence_ratio 的 ID 重新补回。
    if args.allow_presence_backfill and len(kept_roots) < min(args.max_ids, len(stats)):
        for ratio, cnt, r in stats:
            if r in kept_roots:
                continue
            if len(kept_roots) >= args.max_ids:
                break
            kept_roots.append(r)

    root2gid = {r: g for g, r in enumerate(kept_roots)}
    local2global = {}
    for t in ids:
        r = find(t)
        if r in root2gid:
            local2global[t] = root2gid[r]

    dropped = len(cluster_frames) - len(kept_roots)
    print(f"[Stage 2] 合并 {merged} 次; 合并后簇数 {len(cluster_frames)}, "
          f"出场过滤剔除 {dropped} 个低出场/背景 ID")
    print(f"[Stage 2] 最终全局 ID 数: {len(kept_roots)} "
          f"(出场占比阈值 {args.min_presence_ratio:.0%})")
    return local2global


# ============================================================
#  Stage 3: 渲染可视化视频 + 导出 MOT 结果
# ============================================================
def assign_team_colors(local2global, tracklets):
    gid_col = {}
    tmp = defaultdict(lambda: [np.zeros(256, np.float32), 0])
    for t, g in local2global.items():
        c = _finalize_tracklet(tracklets[t])[1]
        tmp[g][0] += c
        tmp[g][1] += 1
    gids = list(tmp.keys())
    if len(gids) < 3:
        return {g: (0, 200, 0) for g in gids}
    mat = np.stack([tmp[g][0] / max(tmp[g][1], 1) for g in gids]).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(mat, 3, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    palette = [(0, 0, 230), (230, 120, 0), (0, 200, 200)]
    for g, lb in zip(gids, labels.flatten()):
        gid_col[g] = palette[int(lb) % 3]
    return gid_col


def stage3_render_and_mot(args, detections, local2global, tracklets):
    print("[Stage 3] 渲染可视化视频 + 导出 MOT ...")
    gid_col = assign_team_colors(local2global, tracklets)

    by_frame = defaultdict(list)
    global_frame_keys = set()
    for (f, tid, x, y, w, h, c) in detections:
        if tid in local2global:            # 仅渲染保留的 ID
            key = (int(f), int(local2global[tid]))
            if key in global_frame_keys:
                raise RuntimeError(f"同帧 global_id 重复，拒绝导出 MOT: frame_proc={key[0]}, global_id={key[1]}")
            global_frame_keys.add(key)
            by_frame[f].append((tid, x, y, w, h, c))

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = processed_fps(fps, args.vid_stride)

    vpath = os.path.join(args.outdir, "tracking_vis.mp4")
    writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"),
                             out_fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频: {vpath}")
    mot_lines = []

    proc_idx, read_idx = -1, -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        read_idx += 1
        if not is_sampled_raw_frame(read_idx, args.vid_stride):
            continue
        proc_idx += 1
        for (tid, x, y, w, h, c) in by_frame.get(proc_idx, []):
            g = local2global[tid]
            col = gid_col.get(g, (0, 200, 0))
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), col, 2)
            cv2.putText(frame, f"ID {g}", (int(x), int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            mot_lines.append(
                f"{proc_idx+1},{g},{x:.1f},{y:.1f},{w:.1f},{h:.1f},{c:.3f},-1,-1,-1")
        writer.write(frame)
    cap.release()
    writer.release()

    mot_path = os.path.join(args.outdir, "tracking_mot.txt")
    with open(mot_path, "w") as f:
        f.write("\n".join(mot_lines))
    print(f"[Stage 3] 可视化视频: {vpath}")
    print(f"[Stage 3] MOT 结果:   {mot_path}")
    return fps, out_fps


# ============================================================
#  Stage 4: 事件检测与切片
# ============================================================
def _interp_series(pos, total, max_gap):
    x = np.full(total, np.nan)
    y = np.full(total, np.nan)
    for f, (cx, cy, _) in pos.items():
        if 0 <= f < total:
            x[f], y[f] = cx, cy
    valid = ~np.isnan(x)
    if valid.sum() < 2:
        return None, None, None
    idx = np.arange(total)
    x = np.interp(idx, idx[valid], x[valid])
    y = np.interp(idx, idx[valid], y[valid])
    reliable = np.zeros(total, bool)
    vi = np.where(valid)[0]
    for k in range(len(vi) - 1):
        if vi[k + 1] - vi[k] <= max_gap:
            reliable[vi[k]:vi[k + 1] + 1] = True
    reliable[vi] = True
    return x, y, reliable


def _smooth(a, w=5):
    if len(a) < w:
        return a
    k = np.ones(w) / w
    return np.convolve(a, k, mode="same")


def stage4_events(args, detections, ball_pos, local2global,
                  total_frames, fps, out_fps):
    print("[Stage 4] 事件检测与切片 ...")
    # Stage 1 的所有输出已经在处理帧域，Stage 4 直接沿用其长度和键。
    proc_total = total_frames
    _validate_proc_frame_domain(proc_total, detections, ball_pos)
    expected_out_fps = processed_fps(fps, args.vid_stride)
    if not math.isclose(out_fps, expected_out_fps, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"处理帧率不一致: out_fps={out_fps}, 应为 {expected_out_fps} "
            f"(raw_fps={fps}, vid_stride={args.vid_stride})"
        )

    bx, by, reliable = _interp_series(ball_pos, proc_total, args.ball_max_gap)

    score = np.zeros(proc_total)
    etype = ["关键动作"] * proc_total
    if bx is not None:
        bx, by = _smooth(bx), _smooth(by)
        vx, vy = np.gradient(bx) * out_fps, np.gradient(by) * out_fps
        speed = np.hypot(vx, vy)
        ax, ay = np.gradient(vx), np.gradient(vy)
        accel = np.hypot(ax, ay)
        ang = np.zeros(proc_total)
        for i in range(1, proc_total):
            v1 = np.array([vx[i - 1], vy[i - 1]])
            v2 = np.array([vx[i], vy[i]])
            if np.linalg.norm(v1) > 1 and np.linalg.norm(v2) > 1:
                cosv = np.clip(np.dot(v1, v2) /
                               (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
                ang[i] = np.degrees(np.arccos(cosv))

        def z(a):
            s = a.std()
            return (a - a.mean()) / s if s > 1e-6 else a * 0
        score = z(accel) + z(ang * speed)
        score[~reliable] = -1e9
        acc95 = np.percentile(accel[reliable], 95) if reliable.any() else 1e9
        for i in range(proc_total):
            if reliable[i]:
                etype[i] = "射门_大力踢球" if accel[i] > acc95 \
                    else ("传球_解围_方向突变" if ang[i] > 90 else "关键动作")

    # 球员速度信号: 仅使用保留的球员 ID
    ptraj = defaultdict(dict)
    for (f, tid, x, y, w, h, c) in detections:
        if tid not in local2global:
            continue
        ptraj[tid][f] = (x + w / 2, y + h / 2)
    pspeed = np.zeros(proc_total)
    for tid, tr in ptraj.items():
        fs = sorted(tr)
        for k in range(1, len(fs)):
            df = fs[k] - fs[k - 1]
            if df == 0 or df > 3:
                continue
            (x0, y0), (x1, y1) = tr[fs[k - 1]], tr[fs[k]]
            sp = np.hypot(x1 - x0, y1 - y0) / df * out_fps
            pspeed[fs[k]] = max(pspeed[fs[k]], sp)
    pspeed = _smooth(pspeed)
    ps = pspeed.std()
    if ps > 1e-6:
        score = score + 0.5 * (pspeed - pspeed.mean()) / ps

    em = int(args.edge_margin * out_fps)
    if em > 0:
        score[:em] = -1e9
        score[-em:] = -1e9

    min_sep = int(args.event_min_gap * out_fps)
    valid_score = score[score > -1e8]
    thr = np.percentile(valid_score, args.event_percentile) \
        if valid_score.size else 1e9
    order = np.argsort(score)[::-1]
    events, occupied = [], np.zeros(proc_total, bool)
    for i in order:
        if score[i] < thr or score[i] <= -1e8:
            break
        if occupied[max(0, i - min_sep):i + min_sep].any():
            continue
        occupied[max(0, i - min_sep):i + min_sep] = True
        events.append(int(i))
    events.sort()
    print(f"[Stage 4] 检测到事件 {len(events)} 个")

    def ts(fp):
        sec = fp / out_fps
        return f"{int(sec//60):02d}:{sec%60:06.3f}"

    rows = []
    for e in events:
        s, t, end_exclusive = clip_bounds(
            e, proc_total, out_fps, args.pre_sec, args.post_sec
        )
        rows.append({
            "event_id": len(rows) + 1,
            "event_frame_proc": int(e),
            "event_time": ts(e),
            "clip_start_time": ts(s),
            "clip_end_time": ts(end_exclusive),
            "event_type": etype[e],
            "score": round(float(score[e]), 3),
            "confidence_note": f"综合代理信号得分 {score[e]:.2f}(球加速度/方向突变+球员速度突变)",
            "start_frame_proc": int(s),
            "end_frame_proc": int(t),
        })

    idx_csv = os.path.join(args.outdir, "event_index.csv")
    with open(idx_csv, "w", encoding="utf-8-sig") as f:
        cols = ["event_id", "event_time", "clip_start_time", "clip_end_time",
                "event_type", "score", "confidence_note"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    with open(os.path.join(args.outdir, "event_index.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[Stage 4] 事件索引表: {idx_csv}")

    # ---- 按事件类型分层抽样, 保证各类动作均衡 ----
    n = min(args.n_clips, len(rows))
    if n == 0:
        print("[Stage 4] 无事件可切片。")
        return

    groups = defaultdict(list)
    for r in rows:
        groups[r["event_type"]].append(r)
    for t in groups:                       # 类内按得分降序
        groups[t].sort(key=lambda r: r["score"], reverse=True)

    types = sorted(groups.keys())
    sampled, ti = [], 0
    while len(sampled) < n and any(groups[t] for t in types):
        t = types[ti % len(types)]         # 各类轮流取
        ti += 1
        if groups[t]:
            sampled.append(groups[t].pop(0))
    sampled.sort(key=lambda r: r["event_id"])

    dist = defaultdict(int)
    for r in sampled:
        dist[r["event_type"]] += 1
    print("[Stage 4] 切片类型分布: " +
          ", ".join(f"{k}:{v}" for k, v in dist.items()))

    clip_dir = os.path.join(args.outdir, "clips")
    os.makedirs(clip_dir, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writers = {}
    for r in sampled:
        fname = f"event_{r['event_id']:03d}_{safe_name(r['event_type'])}.mp4"
        p = os.path.join(clip_dir, fname)
        wr = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
        if not wr.isOpened():
            print(f"    [警告] 无法创建切片 {p}, 跳过")
            continue
        writers[r["event_id"]] = (wr, r["start_frame_proc"], r["end_frame_proc"])

    proc_idx, read_idx = -1, -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        read_idx += 1
        if not is_sampled_raw_frame(read_idx, args.vid_stride):
            continue
        proc_idx += 1
        for eid, (wr, s, t) in writers.items():
            if s <= proc_idx <= t:
                wr.write(frame)
    cap.release()
    for wr, _, _ in writers.values():
        wr.release()
    print(f"[Stage 4] 已导出 {len(writers)} 个抽样切片到 {clip_dir}")


def clip_bounds(event_frame, proc_total, out_fps, pre_sec, post_sec):
    """Return inclusive write bounds plus the exclusive timestamp boundary."""
    start = max(0, int(event_frame) - int(pre_sec * out_fps))
    end_exclusive = min(int(proc_total), int(event_frame) + int(post_sec * out_fps))
    end_inclusive = max(start, end_exclusive - 1)
    return start, end_inclusive, end_exclusive


def validate_args(ap, args):
    if args.vid_stride < 1:
        ap.error("--vid_stride 必须 >= 1")
    if args.reid_stride < 1:
        ap.error("--reid_stride 必须 >= 1")
    if args.reid_interval_sec is not None and args.reid_interval_sec < 0:
        ap.error("--reid_interval_sec 必须 >= 0（0 表示使用 --reid_stride）")
    if args.max_ids < 1:
        ap.error("--max_ids 必须 >= 1")
    if args.min_track_frames < 1:
        ap.error("--min_track_frames 必须 >= 1")
    if not 0 <= args.min_presence_ratio <= 1:
        ap.error("--min_presence_ratio 必须在 0..1")
    if not 0 <= args.conf <= 1:
        ap.error("--conf 必须在 0..1")
    if not 0 <= args.event_percentile <= 100:
        ap.error("--event_percentile 必须在 0..100")
    if args.pre_sec < 0 or args.post_sec < 0:
        ap.error("--pre_sec/--post_sec 必须 >= 0")
    if args.pre_sec + args.post_sec <= 0:
        ap.error("--pre_sec + --post_sec 必须 > 0")
    if args.event_min_gap < 0 or args.edge_margin < 0:
        ap.error("--event_min_gap/--edge_margin 必须 >= 0")
    if args.ball_max_gap < 0:
        ap.error("--ball_max_gap 必须 >= 0")
    if args.n_clips < 0:
        ap.error("--n_clips 必须 >= 0")
    if args.imgsz < 1:
        ap.error("--imgsz 必须 >= 1")


# ============================================================
#  主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--weights", default="yolov8x.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--vid_stride", type=int, default=1)
    ap.add_argument("--reid_stride", type=int, default=10)
    ap.add_argument("--reid_interval_sec", type=float, default=0.33,
                    help="按真实时间配置 ReID 间隔，默认 0.33 秒；0 表示使用 reid_stride")
    # 全局重关联 + 出场过滤
    ap.add_argument("--max_ids", type=int, default=35,
                    help="最终全局 ID 上限")
    ap.add_argument("--min_track_frames", type=int, default=10,
                    help="合并前剔除出场少于该帧数的碎片 tracklet")
    ap.add_argument("--min_presence_ratio", type=float, default=0.01,
                    help="合并后各 ID 出场占比低于此值视为背景人员剔除")
    ap.add_argument("--allow_presence_backfill", action="store_true",
                    help="允许用未达出场阈值的 ID 补齐到 max_ids（默认关闭）")
    ap.add_argument("--merge_floor", type=float, default=0.10)
    ap.add_argument("--wa", type=float, default=0.6)
    ap.add_argument("--wc", type=float, default=0.4)
    ap.add_argument("--color_min", type=float, default=0.15)
    # 事件
    ap.add_argument("--pre_sec", type=float, default=3.0)
    ap.add_argument("--post_sec", type=float, default=2.0)
    ap.add_argument("--event_percentile", type=float, default=92.0)
    ap.add_argument("--event_min_gap", type=float, default=2.0)
    ap.add_argument("--edge_margin", type=float, default=1.0)
    ap.add_argument("--ball_max_gap", type=int, default=30)
    ap.add_argument("--n_clips", type=int, default=20)
    args = ap.parse_args()
    validate_args(ap, args)

    os.makedirs(args.outdir, exist_ok=True)
    device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    if not torch.cuda.is_available():
        device, args.device = "cpu", "cpu"
    print(f"使用设备: {device}")

    reid = ReIDExtractor(device=device)
    detections, ball_pos, tracklets, total_frames = stage1_detect_track(args, reid)
    local2global = stage2_global_reassoc(tracklets, total_frames, args)
    meta_cap = cv2.VideoCapture(args.video)
    fps = meta_cap.get(cv2.CAP_PROP_FPS) or 30.0
    meta_cap.release()
    out_fps = processed_fps(fps, args.vid_stride)
    fps, out_fps = stage3_render_and_mot(args, detections, local2global, tracklets)
    stage4_events(args, detections, ball_pos, local2global,
                  total_frames, fps, out_fps)
    print("\n全部完成。交付物见:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
