"""Write a visual sheet: trunk original / light / heavy. Fragment crops are display-only."""
import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reid_model"))
from augmentations import apply_fragment_view


def load_rgb(path):
    arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError(path)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def panel(image, size=(128, 256)):
    return image.convert("RGB").resize(size, Image.Resampling.BICUBIC)


def caption(image, text):
    canvas = Image.new("RGB", (image.width, image.height + 22), (20, 20, 20))
    canvas.paste(image, (0, 22))
    ImageDraw.Draw(canvas).text((4, 4), text, fill=(240, 240, 240))
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--crop-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pairs", default="129:5,124:9,150:25,142:181,198:307")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    train = [r for r in csv.DictReader(a.train_manifest.open(encoding="utf-8")) if r["split"] == "train"]
    by_train = {}
    for r in train:
        by_train.setdefault(int(r["algorithm_id"]), []).append(r)
    by_crop = {}
    for r in csv.DictReader(a.crop_manifest.open(encoding="utf-8")):
        by_crop.setdefault(int(r["algorithm_id"]), []).append(r)
    rows = []
    for spec in a.pairs.split(","):
        trunk_s, frag_s = spec.split(":")
        trunk, frag = int(trunk_s), int(frag_s)
        if trunk not in by_train:
            continue
        seed = sorted(by_train[trunk], key=lambda r: int(r["frame"]))[len(by_train[trunk]) // 2]
        root = a.train_manifest.parent
        src = Path(seed["image"])
        if not src.is_absolute():
            src = root / src
        original = load_rgb(src)
        light = apply_fragment_view(original, "light", random.Random(trunk * 10 + 1))
        heavy = apply_fragment_view(original, "heavy", random.Random(trunk * 10 + 2))
        cells = [
            caption(panel(original), f"trunk {trunk} original"),
            caption(panel(light), f"trunk {trunk} light"),
            caption(panel(heavy), f"trunk {trunk} heavy"),
        ]
        if frag in by_crop:
            frag_row = sorted(by_crop[frag], key=lambda r: int(r["frame"]))[len(by_crop[frag]) // 2]
            cells.append(caption(panel(load_rgb(frag_row["image"])), f"fragment {frag} (not trained)"))
        strip = Image.new("RGB", (sum(c.width for c in cells) + 8 * (len(cells) - 1), cells[0].height), (8, 8, 8))
        x = 0
        for cell in cells:
            strip.paste(cell, (x, 0))
            x += cell.width + 8
        rows.append(strip)
        strip.save(a.output / f"trunk{trunk}_vs_{frag}.jpg", quality=92)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows) + 8 * (len(rows) - 1)), (8, 8, 8))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + 8
    sheet.save(a.output / "comparison.jpg", quality=92)
    print(str(a.output / "comparison.jpg"))


if __name__ == "__main__":
    main()
