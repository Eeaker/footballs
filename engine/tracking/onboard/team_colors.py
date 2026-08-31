from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .models import TeamCluster, TeamColorResult
from .video_health import read_video_metadata, uniform_frame_indices
from tracking_lib.field_geometry import FieldGeometryProvider
from tracking_lib.field_filter import turf_support_score
from tracking_lib.team_features import aggregate_features, representative_hsv, suggested_color_name, torso_feature


def kmeans_labels(features: np.ndarray, k: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """使用固定随机种子的 OpenCV kmeans 聚类，保证适配结果可复现。"""
    if len(features) < k:
        raise ValueError(f"样本数 {len(features)} 小于聚类数 {k}")
    cv2.setRNGSeed(seed)
    _, labels, centers = cv2.kmeans(
        features.astype(np.float32), k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4),
        10, cv2.KMEANS_PP_CENTERS,
    )
    return labels.reshape(-1), centers


def silhouette_score(features: np.ndarray, labels: np.ndarray, max_samples: int = 800) -> float:
    """计算轮廓系数；大样本时确定性抽样以限制内存。"""
    if len(set(labels.tolist())) < 2 or len(features) < 3:
        return -1.0
    if len(features) > max_samples:
        indices = np.linspace(0, len(features) - 1, max_samples).round().astype(int)
        features, labels = features[indices], labels[indices]
    distance = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    values = []
    for i, own in enumerate(labels):
        same = labels == own
        a = float(distance[i, same].sum() / max(int(same.sum()) - 1, 1))
        b = min(float(distance[i, labels == other].mean()) for other in set(labels.tolist()) if other != own)
        values.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(values))


def resize_with_padding(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """保持比例缩放裁片，避免色板拉伸球员外观。"""
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
    canvas = np.full((height, width, 3), 235, np.uint8)
    y = (height - resized.shape[0]) // 2; x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def build_color_board(crops: list[np.ndarray], cluster_features: np.ndarray, labels: np.ndarray,
                      centers: np.ndarray, output: str | Path, k: int,
                      quality_scores: np.ndarray) -> tuple[list[list[int]], list[str], list[list[float]]]:
    """用最接近簇中心的球衣裁片生成确认板。"""
    tile_w, tile_h = 180, 205
    canvas = np.full((k * tile_h, 5 * tile_w, 3), 245, np.uint8)
    colors, names, hsv_colors = [], [], []
    for cluster in range(k):
        ids = np.where(labels == cluster)[0]
        ordered = ids[np.argsort(np.linalg.norm(cluster_features[ids] - centers[cluster], axis=1))]
        selected = ordered[:min(4, len(ordered))]
        for column, crop_id in enumerate(selected):
            tile = resize_with_padding(crops[int(crop_id)], tile_w - 12, tile_h - 48)
            canvas[cluster * tile_h + 38:(cluster + 1) * tile_h - 10,
                   column * tile_w + 6:(column + 1) * tile_w - 6] = tile
        hsv = representative_hsv([crops[int(i)] for i in ids])
        swatch = cv2.cvtColor(np.uint8([[np.clip(hsv, 0, 255)]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        name = suggested_color_name(hsv)
        colors.append([int(value) for value in swatch]); names.append(name)
        hsv_colors.append([round(float(value), 2) for value in hsv])
        canvas[cluster * tile_h + 5:cluster * tile_h + 31, 4 * tile_w + 12:5 * tile_w - 12] = swatch
        quality = float(np.mean(quality_scores[ids])) if len(ids) else 0.0
        cv2.putText(canvas, f"Cluster {cluster}  n={len(ids)}  color={name}  field={quality:.2f}",
                    (10, cluster * tile_h + 27), cv2.FONT_HERSHEY_SIMPLEX, .62, (20, 20, 20), 2)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"无法写入队色色板: {output}")
    return colors, names, hsv_colors


def build_clusters(k: int, crops: list[np.ndarray], appearance_features: np.ndarray,
                   cluster_features: np.ndarray, quality_scores: np.ndarray,
                   labels: np.ndarray, centers: np.ndarray, output_board: str | Path) -> list[TeamCluster]:
    """生成可序列化的簇摘要和可视化色板。"""
    colors, names, hsv_colors = build_color_board(
        crops, cluster_features, labels, centers, output_board, k, quality_scores)
    result = []
    for cluster in range(k):
        ids = np.where(labels == cluster)[0]
        appearance_center = aggregate_features([appearance_features[int(i)] for i in ids])
        quality = float(np.mean(quality_scores[ids])) if len(ids) else 0.0
        result.append(TeamCluster(cluster, len(ids), hsv_colors[cluster], colors[cluster],
                                  names[cluster], appearance_center.astype(float).tolist(),
                                  quality, f"cluster_{cluster}_{names[cluster]}"))
    return result


def analyze_team_colors(video_path: str | Path, weights: str | Path, output_board: str | Path,
                        device: str = "0", sample_count: int = 50,
                        detector: Callable | None = None, min_turf_support: float = .15,
                        min_track_turf_ratio: float = .25, min_geometry_ratio: float = .60,
                        field_geometry: dict | None = None,
                        tracker_config: str | Path = "botsort.yaml") -> tuple[TeamColorResult, dict]:
    """执行轨迹级队色分析：先筛场内轨迹，再让每条轨迹贡献一个颜色原型。"""
    from collections import defaultdict

    meta = read_video_metadata(video_path)
    geometry = FieldGeometryProvider(field_geometry)
    window_count = 5
    frames_per_window = max(12, sample_count // window_count)
    centers = uniform_frame_indices(meta.frame_count, window_count, margin_ratio=.06)
    track_rows: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"seen": 0, "geometry": 0, "turf": 0, "features": [], "crops": []}
    )
    if detector is None:
        from ultralytics import YOLO
        model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video_path))
    processed_frames = 0
    for window_id, center in enumerate(centers):
        start = max(0, int(center) - frames_per_window // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        if detector is None:
            model.predictor = None
        for offset in range(frames_per_window):
            ok, frame = cap.read()
            if not ok:
                break
            raw_frame = start + offset
            if detector is None:
                result = model.track(frame, classes=[0], tracker=str(tracker_config), persist=True,
                                     conf=.25, iou=.5, imgsz=1280, device=device, verbose=False)[0]
            else:
                result = detector(frame)
            processed_frames += 1
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            coords = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            if getattr(boxes, "id", None) is None:
                identities = np.arange(len(coords), dtype=int) + offset * 10000
            else:
                identities = boxes.id.int().cpu().numpy()
            for identity, box in zip(identities, coords):
                x1, y1, x2, y2 = map(float, box)
                key = (window_id, int(identity)); row = track_rows[key]; row["seen"] += 1
                foot = ((x1 + x2) / 2.0, y2)
                geometry_ok = geometry.contains(raw_frame, foot)
                support = turf_support_score(frame, (x1, y1, x2 - x1, y2 - y1))
                row["geometry"] += int(geometry_ok)
                row["turf"] += int(support >= min_turf_support)
                if not geometry_ok:
                    continue
                item = torso_feature(frame, tuple(box))
                if item:
                    feature, crop = item
                    row["features"].append(feature); row["crops"].append(crop)
    cap.release()

    prototypes, crops, qualities, accepted_details = [], [], [], []
    for key, row in track_rows.items():
        geometry_ratio = row["geometry"] / max(row["seen"], 1)
        turf_ratio = row["turf"] / max(row["seen"], 1)
        accepted = (row["seen"] >= 3 and geometry_ratio >= min_geometry_ratio
                    and turf_ratio >= min_track_turf_ratio and len(row["features"]) >= 2)
        accepted_details.append({"window": key[0], "local_track_id": key[1], "samples": row["seen"],
                                 "geometry_ratio": round(geometry_ratio, 4),
                                 "turf_ratio": round(turf_ratio, 4), "accepted": accepted})
        if not accepted:
            continue
        prototype = aggregate_features(row["features"])
        distances = [float(np.linalg.norm(feature - prototype)) for feature in row["features"]]
        duration_score = min(1.0, row["seen"] / max(frames_per_window * .5, 1.0))
        quality = .45 * geometry_ratio + .40 * turf_ratio + .15 * duration_score
        prototypes.append(prototype); crops.append(row["crops"][int(np.argmin(distances))])
        qualities.append(quality)
    if len(prototypes) < 6:
        raise RuntimeError(f"可用轨迹仅 {len(prototypes)} 条，无法构建自动三簇；请检查场地ROI和投票阈值")
    appearance_matrix = np.asarray(prototypes, np.float32)
    quality_matrix = np.asarray(qualities, np.float32)
    # 正式重关联只具备外观原型，因此聚类也只使用同一特征空间；场地、脚底
    # 草地和轨迹持续性在此之前完成筛选，避免训练/应用特征不一致。
    cluster_matrix = appearance_matrix
    trials = {k: kmeans_labels(cluster_matrix, k) for k in (2, 3)}
    scores = {k: silhouette_score(cluster_matrix, trials[k][0]) for k in trials}
    # 工程口径固定三簇：三个外观簇彼此形成重关联屏障，不额外解释或丢弃某簇。
    recommended = 3
    labels, centers = trials[recommended]
    clusters = build_clusters(recommended, crops, appearance_matrix, cluster_matrix,
                              quality_matrix, labels, centers, output_board)
    result = TeamColorResult(processed_frames, len(crops), scores[2], scores[3], recommended, recommended,
                             clusters, str(Path(output_board).resolve()))
    return result, {"trials": trials, "appearance_features": appearance_matrix,
                    "cluster_features": cluster_matrix, "quality_scores": quality_matrix, "crops": crops,
                    "track_filter": accepted_details}


def select_color_clustering(result: TeamColorResult, analysis: dict, selected_k: int,
                            output_board: str | Path) -> TeamColorResult:
    """按指定 K 重建色板和簇摘要；新视频适配流程固定传入 K=3。"""
    if selected_k not in (2, 3):
        raise ValueError("球衣簇数只能选择 2 或 3")
    labels, centers = analysis["trials"][selected_k]
    result.selected_k = selected_k
    result.clusters = build_clusters(selected_k, analysis["crops"], analysis["appearance_features"],
                                     analysis["cluster_features"], analysis["quality_scores"],
                                     labels, centers, output_board)
    result.board_path = str(Path(output_board).resolve())
    return result
