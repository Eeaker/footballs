"""Render the final match analysis acceptance demo with honest ball and possession states."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


BallVisual = tuple[float | None, float | None, str]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_events(pass_events: list[dict[str, str]], sample_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_id = {int(row["pass_id"]): row for row in pass_events}
    selected = []
    for sample in sample_rows:
        pass_id = int(sample["pass_id"])
        if pass_id not in by_id:
            raise ValueError(f"sample pass_id missing from pass events: {pass_id}")
        selected.append(by_id[pass_id])
    return selected


def read_observed_ball(path: Path) -> dict[int, tuple[float, float]]:
    return {
        int(row["frame_proc"]): (float(row["ball_x_px"]), float(row["ball_y_px"]))
        for row in read_csv(path)
        if str(row.get("observed", "1")).strip().lower() in {"1", "true", "yes"}
    }


def build_ball_visual_track(
    observed: dict[int, tuple[float, float]], max_interpolation_gap_frames: int,
) -> dict[int, tuple[float, float, str]]:
    """Fill only short internal gaps and label them as visualization interpolation."""
    if max_interpolation_gap_frames < 0:
        raise ValueError("max_interpolation_gap_frames must be nonnegative")
    track = {frame: (xy[0], xy[1], "observed") for frame, xy in observed.items()}
    frames = sorted(observed)
    for left, right in zip(frames, frames[1:]):
        missing = right - left - 1
        if missing <= 0 or missing > max_interpolation_gap_frames:
            continue
        x0, y0 = observed[left]
        x1, y1 = observed[right]
        span = right - left
        for frame in range(left + 1, right):
            alpha = (frame - left) / span
            track[frame] = (x0 + (x1 - x0) * alpha, y0 + (y1 - y0) * alpha, "visual_interp")
    return track


def resolve_ball_state(
    frame: int, track: dict[int, tuple[float, float, str]],
    last_seen: tuple[int, float, float] | None, hold_frames: int,
) -> BallVisual:
    current = track.get(frame)
    if current is not None:
        return current
    if last_seen is not None and 0 < frame - last_seen[0] <= hold_frames:
        return last_seen[1], last_seen[2], "last_seen"
    return None, None, "unobserved"


def build_possession_index(intervals: list[dict[str, str]]) -> dict[int, tuple[int, str]]:
    """Index only frames from confirmation onward; pre-confirmation frames remain unknown."""
    index: dict[int, tuple[int, str]] = {}
    for row in intervals:
        start = int(row["confirmed_frame_proc"])
        end = int(row["end_frame_proc"])
        holder = (int(row["global_id"]), row["team_id"])
        for frame in range(start, end + 1):
            index[frame] = holder
    return index


def read_mot(path: Path) -> dict[int, list[dict[str, Any]]]:
    frames: dict[int, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip().split(",")
            frame = int(fields[0]) - 1
            frames.setdefault(frame, []).append({
                "global_id": int(fields[1]), "x": float(fields[2]), "y": float(fields[3]),
                "w": float(fields[4]), "h": float(fields[5]), "confidence": float(fields[6]),
            })
    return frames


def read_team_map(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    return {int(row["global_id"]): row["team_id"] for row in read_csv(path)}


def frame_bounds(
    release: int, receive: int, total_frames: int, fps: float,
    before_seconds: float, after_seconds: float,
) -> tuple[int, int]:
    return (
        max(0, release - int(round(before_seconds * fps))),
        min(total_frames, receive + int(round(after_seconds * fps))),
    )


def _box_points(box: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    x1, y1 = int(round(box["x"])), int(round(box["y"]))
    x2, y2 = int(round(box["x"] + box["w"])), int(round(box["y"] + box["h"]))
    foot = (int(round(box["x"] + box["w"] / 2)), y2)
    return (x1, y1), (x2, y2), foot


def _draw_box(frame: np.ndarray, box: dict[str, Any], color: tuple[int, int, int], thickness: int, label: str) -> None:
    p1, p2, foot = _box_points(box)
    cv2.rectangle(frame, p1, p2, color, thickness)
    cv2.circle(frame, foot, 4 if thickness > 1 else 2, color, -1)
    cv2.putText(frame, label, (p1[0], max(88, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2)


def _draw_ball(frame: np.ndarray, state: BallVisual, trail: deque[tuple[int, int]]) -> None:
    x, y, mode = state
    if x is None or y is None:
        return
    point = (int(round(x)), int(round(y)))
    if mode in {"observed", "visual_interp"}:
        trail.append(point)
    if len(trail) >= 2:
        points = np.asarray(trail, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], False, (80, 80, 220), 2, cv2.LINE_AA)
    if mode == "observed":
        cv2.circle(frame, point, 11, (0, 0, 255), -1)
        cv2.circle(frame, point, 13, (255, 255, 255), 2)
    elif mode == "visual_interp":
        cv2.circle(frame, point, 11, (0, 165, 255), -1)
        cv2.circle(frame, point, 13, (255, 255, 255), 2)
    elif mode == "last_seen":
        cv2.circle(frame, point, 12, (170, 170, 170), 2)
        cv2.line(frame, (point[0]-8, point[1]-8), (point[0]+8, point[1]+8), (170, 170, 170), 2)


def _status_color(mode: str) -> tuple[int, int, int]:
    return {
        "observed": (80, 230, 80),
        "visual_interp": (0, 185, 255),
        "last_seen": (190, 190, 190),
        "unobserved": (100, 100, 255),
    }[mode]


def _put_panel(frame: np.ndarray, y1: int, y2: int) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (frame.shape[1], y2), (12, 18, 28), -1)
    cv2.addWeighted(overlay, .88, frame, .12, 0, frame)


def render_demo(args: argparse.Namespace) -> dict[str, Any]:
    if args.outdir.exists():
        raise FileExistsError(f"output directory must not exist: {args.outdir}")
    if args.before_release_seconds < 0 or args.after_receive_seconds < 0:
        raise ValueError("clip context seconds must be nonnegative")

    events = select_events(read_csv(args.pass_events), read_csv(args.sample))
    observed = read_observed_ball(args.ball)
    ball_track = build_ball_visual_track(observed, args.visual_interpolation_gap_frames)
    possession = build_possession_index(read_csv(args.possession_intervals))
    mot = read_mot(args.mot)
    team_map = read_team_map(args.team_map)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    args.outdir.mkdir(parents=True)
    compilation_path = args.outdir / f"task3_acceptance_final_{len(events)}.mp4"
    compilation = cv2.VideoWriter(str(compilation_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not compilation.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create compilation: {compilation_path}")

    manifest_events = []
    try:
        for order, event in enumerate(events, 1):
            pass_id = int(event["pass_id"])
            source_id, target_id = int(event["from_global_id"]), int(event["to_global_id"])
            release, receive = int(event["release_frame_proc"]), int(event["receive_frame_proc"])
            receive_confirmed = int(event["receive_confirmed_frame_proc"])
            start, end = frame_bounds(
                release, receive, total_frames, fps,
                args.before_release_seconds, args.after_receive_seconds,
            )
            clip_name = f"candidate_{order:02d}_pass_{pass_id:03d}.mp4"
            writer = cv2.VideoWriter(str(args.outdir / clip_name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"cannot create clip: {clip_name}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            last_seen: tuple[int, float, float] | None = None
            trail: deque[tuple[int, int]] = deque(maxlen=12)
            mode_counts: Counter[str] = Counter()
            possession_frames = 0
            written = 0
            try:
                for frame_index in range(start, end):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    boxes = mot.get(frame_index, [])
                    by_id = {int(box["global_id"]): box for box in boxes}
                    holder = possession.get(frame_index)
                    holder_id = holder[0] if holder else None

                    for box in boxes:
                        gid = int(box["global_id"])
                        team = team_map.get(gid, "team_?")
                        _draw_box(frame, box, (105, 105, 105), 1, f"ID {gid} {team}")
                    if source_id in by_id:
                        _draw_box(frame, by_id[source_id], (0, 165, 255), 3, f"FROM ID {source_id}")
                    if target_id in by_id:
                        _draw_box(frame, by_id[target_id], (60, 220, 60), 3, f"TO ID {target_id}")
                    if holder_id is not None and holder_id in by_id:
                        possession_frames += 1
                        _draw_box(frame, by_id[holder_id], (255, 220, 0), 4, f"POSSESSION ID {holder_id}")

                    if frame_index in ball_track:
                        x, y, mode = ball_track[frame_index]
                        last_seen = (frame_index, x, y)
                        ball_state: BallVisual = (x, y, mode)
                    else:
                        ball_state = resolve_ball_state(
                            frame_index, ball_track, last_seen, args.last_seen_hold_frames,
                        )
                    mode_counts[ball_state[2]] += 1
                    _draw_ball(frame, ball_state, trail)

                    in_transfer = release <= frame_index <= receive_confirmed
                    if in_transfer and source_id in by_id and target_id in by_id:
                        _, _, source_foot = _box_points(by_id[source_id])
                        _, _, target_foot = _box_points(by_id[target_id])
                        cv2.arrowedLine(frame, source_foot, target_foot, (255, 80, 255), 4, cv2.LINE_AA, tipLength=.12)

                    _put_panel(frame, 0, 78)
                    cv2.putText(frame, f"TASK 3 ACCEPTANCE ROBOT   CANDIDATE {order}/{len(events)}   PASS ID {pass_id:03d}",
                                (18, 29), cv2.FONT_HERSHEY_SIMPLEX, .72, (255, 255, 255), 2)
                    cv2.putText(frame, f"ACTIVE DIRECTED PASS CANDIDATE   ID {source_id} -> {target_id}   {float(event['distance_m']):.3f} m   HUMAN REVIEW REQUIRED",
                                (18, 59), cv2.FONT_HERSHEY_SIMPLEX, .58, (210, 230, 255), 2)

                    _put_panel(frame, height - 84, height)
                    ball_mode = ball_state[2].upper().replace("_", " ")
                    cv2.putText(frame, f"BALL: {ball_mode}", (18, height - 51), cv2.FONT_HERSHEY_SIMPLEX, .65,
                                _status_color(ball_state[2]), 2)
                    if in_transfer:
                        possession_text = f"POSSESSION: IN TRANSFER  {source_id} -> {target_id}"
                        possession_color = (255, 80, 255)
                    elif holder:
                        possession_text = f"POSSESSION: ID {holder[0]}  {holder[1]}  CONFIRMED"
                        possession_color = (255, 220, 0)
                    else:
                        possession_text = "POSSESSION: UNCONFIRMED"
                        possession_color = (170, 170, 170)
                    cv2.putText(frame, possession_text, (300, height - 51), cv2.FONT_HERSHEY_SIMPLEX, .65,
                                possession_color, 2)
                    cv2.putText(frame, "RED=observed  ORANGE=visual interpolation  GRAY=last seen  no marker=unobserved",
                                (18, height - 20), cv2.FONT_HERSHEY_SIMPLEX, .52, (205, 205, 205), 1)

                    writer.write(frame)
                    compilation.write(frame)
                    written += 1
            finally:
                writer.release()
            manifest_events.append({
                "sample_order": order, "pass_id": pass_id, "clip_file": clip_name,
                "from_global_id": source_id, "to_global_id": target_id,
                "release_frame_proc": release, "receive_frame_proc": receive,
                "receive_confirmed_frame_proc": receive_confirmed,
                "distance_m": float(event["distance_m"]), "written_frames": written,
                "ball_visual_mode_frames": dict(mode_counts),
                "frames_with_confirmed_possession_box": possession_frames,
            })
    finally:
        compilation.release()
        capture.release()

    manifest = {
        "schema_version": "task3-final-demo-v1",
        "status": "pending_human_review",
        "honesty_policy": {
            "observed": "raw detector observation",
            "visual_interp": f"linear visualization only; internal gaps <= {args.visual_interpolation_gap_frames} frames",
            "last_seen": f"hollow marker retained <= {args.last_seen_hold_frames} frames",
            "unobserved": "no ball position is claimed",
            "possession": "shown only from confirmed_frame_proc through end_frame_proc",
        },
        "event_count": len(manifest_events), "compilation_file": compilation_path.name,
        "video": str(args.video.resolve()), "mot": str(args.mot.resolve()),
        "ball": str(args.ball.resolve()), "events": manifest_events,
    }
    (args.outdir / "task3_final_demo_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final match analysis acceptance demo")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--ball", type=Path, required=True)
    parser.add_argument("--possession-intervals", type=Path, required=True)
    parser.add_argument("--pass-events", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--team-map", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--visual-interpolation-gap-frames", type=int, default=12)
    parser.add_argument("--last-seen-hold-frames", type=int, default=12)
    parser.add_argument("--before-release-seconds", type=float, default=1.0)
    parser.add_argument("--after-receive-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = render_demo(args)
    print(json.dumps({
        "output": str(args.outdir.resolve()), "compilation": manifest["compilation_file"],
        "event_count": manifest["event_count"], "status": manifest["status"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
