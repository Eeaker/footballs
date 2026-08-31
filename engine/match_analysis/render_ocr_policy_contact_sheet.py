from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render OCR support frames for manual policy audit")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--statuses", default="confirmed,tentative")
    parser.add_argument("--columns", type=int, default=4)
    return parser.parse_args()


def render(args: argparse.Namespace) -> list[Path]:
    if args.output.exists():
        raise FileExistsError(f"output must not exist: {args.output}")
    args.output.mkdir(parents=True)
    wanted = {value.strip() for value in args.statuses.split(",") if value.strip()}
    with args.results.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] in wanted]

    tiles: list[np.ndarray] = []
    for row in rows:
        gid = int(row["global_id"])
        for frame_value in str(row.get("support_frames") or "").split(";"):
            if not frame_value:
                continue
            frame = int(frame_value)
            crop = args.candidates / f"gid_{gid:03d}" / f"frame_{frame:06d}.jpg"
            image = cv2.imread(str(crop))
            if image is None:
                continue
            canvas = np.full((280, 360, 3), 28, dtype=np.uint8)
            scale = min(340 / image.shape[1], 220 / image.shape[0])
            resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
            x = (360 - resized.shape[1]) // 2
            y = 42 + (220 - resized.shape[0]) // 2
            canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
            colour = (70, 230, 70) if row["status"] == "confirmed" else (0, 190, 255)
            cv2.putText(
                canvas, f"{row['status'].upper()}  GID {gid}  {row['team']} #{row['predicted_number']}",
                (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .52, colour, 1, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, f"frame {frame}  support {row['support_count']}  conf {float(row['confidence']):.3f}",
                (8, 273), cv2.FONT_HERSHEY_SIMPLEX, .46, (230, 230, 230), 1, cv2.LINE_AA,
            )
            tiles.append(canvas)

    page_size = args.columns * 4
    outputs: list[Path] = []
    for page, offset in enumerate(range(0, len(tiles), page_size), 1):
        batch = tiles[offset:offset + page_size]
        rows_count = (len(batch) + args.columns - 1) // args.columns
        sheet = np.full((rows_count * 280, args.columns * 360, 3), 18, dtype=np.uint8)
        for index, tile in enumerate(batch):
            row, column = divmod(index, args.columns)
            sheet[row * 280:(row + 1) * 280, column * 360:(column + 1) * 360] = tile
        path = args.output / f"ocr_policy_audit_{page:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
        outputs.append(path)
    return outputs


def main() -> None:
    outputs = render(parse_args())
    print("\n".join(map(str, outputs)))


if __name__ == "__main__":
    main()
