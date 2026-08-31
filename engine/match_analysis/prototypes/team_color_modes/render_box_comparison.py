from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


DEFAULT_MAIN_FRAMES = {31: 921, 37: 31825, 39: 20957, 41: 23405, 42: 17053}
DEFAULT_EVIDENCE_FRAMES = {
    31: [921, 3212, 14723],
    37: [6238, 31825, 32958],
    39: [17145, 20957, 21291],
    41: [20211, 23405, 24157],
    42: [17053, 20794, 30137],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render auditable before/after team-colour boxes.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size=size)


def load_assignments(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {int(row["global_id"]): row for row in csv.DictReader(handle)}


def load_boxes(path: Path, wanted: dict[int, set[int]]) -> dict[tuple[int, int], tuple[float, ...]]:
    result: dict[tuple[int, int], tuple[float, ...]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            frame = int(float(row[0])) - 1  # MOT is 1-based; OpenCV/manifest are 0-based.
            gid = int(float(row[1]))
            if gid in wanted and frame in wanted[gid]:
                result[(gid, frame)] = tuple(map(float, row[2:7]))
    missing = [(gid, frame) for gid, frames in wanted.items() for frame in frames if (gid, frame) not in result]
    if missing:
        raise RuntimeError(f"Missing MOT boxes: {missing}")
    return result


def read_frames(video: Path, indices: set[int]) -> tuple[dict[int, Image.Image], float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames: dict[int, Image.Image] = {}
    for index in sorted(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {index}")
        frames[index] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def team_colour(team: str) -> tuple[int, int, int]:
    return (255, 196, 0) if team == "team_0" else (40, 155, 255)


def crop_with_context(image: Image.Image, box: tuple[float, ...], target=(260, 220)) -> Image.Image:
    x, y, w, h, _ = box
    cx, cy = x + w / 2, y + h / 2
    span_w = max(w * 5.0, 150)
    span_h = max(h * 3.1, 125)
    left = max(0, int(cx - span_w / 2))
    top = max(0, int(cy - span_h / 2))
    right = min(image.width, int(cx + span_w / 2))
    bottom = min(image.height, int(cy + span_h / 2))
    return image.crop((left, top, right, bottom)).resize(target, Image.Resampling.LANCZOS)


def draw_box(image: Image.Image, box: tuple[float, ...], colour: tuple[int, int, int], label: str) -> None:
    draw = ImageDraw.Draw(image)
    x, y, w, h, _ = box
    x1, y1, x2, y2 = map(round, (x, y, x + w, y + h))
    thickness = 5
    for offset in range(thickness):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=colour)
    label_font = font(22, bold=True)
    label_box = draw.textbbox((0, 0), label, font=label_font)
    label_w = label_box[2] - label_box[0] + 14
    label_h = label_box[3] - label_box[1] + 12
    ly = max(0, y1 - label_h - 5)
    draw.rectangle((x1, ly, x1 + label_w, ly + label_h), fill=colour)
    draw.text((x1 + 7, ly + 3), label, fill=(10, 18, 26), font=label_font)


def make_panel(
    source: Image.Image,
    box: tuple[float, ...],
    title: str,
    subtitle: str,
    colour: tuple[int, int, int],
    box_label: str,
) -> Image.Image:
    canvas = Image.new("RGB", (720, 500), (18, 24, 32))
    full = source.copy()
    draw_box(full, box, colour, box_label)
    full.thumbnail((680, 382), Image.Resampling.LANCZOS)
    canvas.paste(full, (20, 88))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), title, fill=(245, 248, 252), font=font(27, bold=True))
    draw.text((20, 49), subtitle, fill=(180, 194, 208), font=font(18))

    zoom = crop_with_context(source, box)
    # Redraw from the exact crop coordinates by reusing a second contextual crop with
    # a target-size-relative centre marker; the full-frame box remains authoritative.
    zdraw = ImageDraw.Draw(zoom)
    zdraw.rectangle((2, 2, zoom.width - 3, zoom.height - 3), outline=colour, width=5)
    zdraw.text((10, 8), "局部放大", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0), font=font(18, bold=True))
    canvas.paste(zoom, (430, 235))
    return canvas


def render_main(
    output: Path,
    frames: dict[int, Image.Image],
    boxes: dict[tuple[int, int], tuple[float, ...]],
    assignments: dict[int, dict[str, str]],
    fps: float,
) -> None:
    row_height = 550
    sheet = Image.new("RGB", (1480, 110 + row_height * len(DEFAULT_MAIN_FRAMES)), (9, 14, 21))
    draw = ImageDraw.Draw(sheet)
    draw.text((28, 18), "球队颜色过滤：同帧、同目标框对比", fill=(248, 250, 252), font=font(34, bold=True))
    draw.text((28, 65), "左：原固定 K=2 强制归队　　右：保守异常过滤（取消球队标签，但不删除检测框）", fill=(180, 194, 208), font=font(21))
    for row, (gid, frame_index) in enumerate(DEFAULT_MAIN_FRAMES.items()):
        a = assignments[gid]
        box = boxes[(gid, frame_index)]
        distance = float(a["cluster_medoid_distance"])
        threshold = float(a["cluster_outlier_threshold"])
        is_outlier = int(a["robust_colour_outlier"]) == 1
        time_s = frame_index / fps
        common = f"GID {gid} · 帧 {frame_index} · {time_s:.1f}s · 距离 {distance:.4f} / 阈值 {threshold:.4f}"
        left = make_panel(
            frames[frame_index], box,
            f"旧逻辑：{a['fixed_k2_cluster']}", common,
            team_colour(a["fixed_k2_cluster"]), f"GID {gid} / {a['fixed_k2_cluster']}"
        )
        if is_outlier:
            right_title = "新逻辑：UNASSIGNED（不计入球队）"
            right_colour = (255, 78, 72)
            right_label = f"GID {gid} / UNASSIGNED"
        else:
            right_title = f"新逻辑：{a['fixed_k2_cluster']}（保留）"
            right_colour = (56, 210, 118)
            right_label = f"GID {gid} / KEPT"
        right = make_panel(frames[frame_index], box, right_title, common, right_colour, right_label)
        y = 110 + row * row_height
        sheet.paste(left, (20, y))
        sheet.paste(right, (740, y))
        verdict = "保留：未越阈值" if not is_outlier else "过滤球队标签：已越阈值"
        draw.text((40, y + 505), verdict, fill=right_colour, font=font(22, bold=True))
    sheet.save(output, quality=94, subsampling=0)


def render_evidence(
    output: Path,
    frames: dict[int, Image.Image],
    boxes: dict[tuple[int, int], tuple[float, ...]],
    assignments: dict[int, dict[str, str]],
    fps: float,
) -> None:
    cell_w, cell_h = 460, 330
    sheet = Image.new("RGB", (60 + 3 * cell_w, 105 + len(DEFAULT_EVIDENCE_FRAMES) * cell_h), (10, 15, 22))
    draw = ImageDraw.Draw(sheet)
    draw.text((25, 16), "多帧核验（采样自原身份拼图帧）", fill=(248, 250, 252), font=font(32, bold=True))
    draw.text((25, 59), "绿色：正常对照保留；红色：仅取消球队归属，MOT 框仍在", fill=(180, 194, 208), font=font(20))
    for row, (gid, indices) in enumerate(DEFAULT_EVIDENCE_FRAMES.items()):
        a = assignments[gid]
        outlier = int(a["robust_colour_outlier"]) == 1
        colour = (255, 78, 72) if outlier else (56, 210, 118)
        for col, frame_index in enumerate(indices):
            frame = frames[frame_index].copy()
            box = boxes[(gid, frame_index)]
            draw_box(frame, box, colour, f"GID {gid}")
            zoom = crop_with_context(frame, box, target=(420, 230))
            cell = Image.new("RGB", (cell_w - 10, cell_h - 10), (20, 27, 36))
            cell.paste(zoom, (15, 54))
            cd = ImageDraw.Draw(cell)
            status = "不归队" if outlier else "保留"
            cd.text((15, 10), f"GID {gid} · {status}", fill=colour, font=font(22, bold=True))
            cd.text((345, 14), f"{frame_index / fps:.1f}s", fill=(210, 220, 230), font=font(17), anchor="ra")
            cd.text((15, 289), f"d={float(a['cluster_medoid_distance']):.4f}  T={float(a['cluster_outlier_threshold']):.4f}", fill=(180, 194, 208), font=font(17))
            sheet.paste(cell, (30 + col * cell_w, 105 + row * cell_h))
    sheet.save(output, quality=94, subsampling=0)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    assignments = load_assignments(args.assignments)
    wanted = {gid: set(indices) | {DEFAULT_MAIN_FRAMES[gid]} for gid, indices in DEFAULT_EVIDENCE_FRAMES.items()}
    boxes = load_boxes(args.mot, wanted)
    all_frames = {index for indices in wanted.values() for index in indices}
    frames, fps = read_frames(args.video, all_frames)
    render_main(args.output_dir / "team_colour_filtering_box_comparison.jpg", frames, boxes, assignments, fps)
    render_evidence(args.output_dir / "team_colour_filtering_multiframe_evidence.jpg", frames, boxes, assignments, fps)
    audit = {
        "video": str(args.video), "mot": str(args.mot), "fps": fps,
        "main_frames": DEFAULT_MAIN_FRAMES, "evidence_frames": DEFAULT_EVIDENCE_FRAMES,
        "note": "MOT frame numbers are converted from 1-based to 0-based for OpenCV.",
    }
    (args.output_dir / "selected_frames.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
