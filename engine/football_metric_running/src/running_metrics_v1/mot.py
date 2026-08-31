"""MOT parsing and quality inspection for the unified processed-frame domain."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MotDetection:
    proc_idx: int
    global_id: int
    x: float
    y: float
    w: float
    h: float
    confidence: float

    @property
    def foot_x(self) -> float:
        return self.x + self.w / 2.0

    @property
    def foot_y(self) -> float:
        return self.y + self.h


def read_mot(path: str | Path) -> list[MotDetection]:
    """Read Stage 3 MOT rows; column 1 is one-based processed frame."""
    records: list[MotDetection] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip().split(",")
            if len(fields) < 7:
                raise ValueError(f"invalid MOT row {line_number}: expected >=7 columns")
            frame_mot = int(fields[0])
            if frame_mot < 1:
                raise ValueError(f"invalid MOT frame at row {line_number}: {frame_mot}")
            records.append(
                MotDetection(
                    proc_idx=frame_mot - 1,
                    global_id=int(fields[1]),
                    x=float(fields[2]),
                    y=float(fields[3]),
                    w=float(fields[4]),
                    h=float(fields[5]),
                    confidence=float(fields[6]),
                )
            )
    return records


def group_by_identity_frame(
    records: list[MotDetection],
) -> dict[int, dict[int, list[MotDetection]]]:
    grouped: dict[int, dict[int, list[MotDetection]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        grouped[record.global_id][record.proc_idx].append(record)
    return {gid: dict(frames) for gid, frames in grouped.items()}


def inspect_records(records: list[MotDetection], total_proc_frames: int | None = None) -> dict:
    grouped = group_by_identity_frame(records)
    per_frame = Counter(r.proc_idx for r in records)
    duplicate_keys = [
        (gid, frame, len(items))
        for gid, frames in grouped.items()
        for frame, items in frames.items()
        if len(items) > 1
    ]
    if total_proc_frames is None:
        total_proc_frames = max(per_frame, default=-1) + 1

    identities = []
    for gid, frames in sorted(grouped.items()):
        indices = sorted(frames)
        gaps = [b - a for a, b in zip(indices, indices[1:])]
        span = indices[-1] - indices[0] + 1 if indices else 0
        identities.append(
            {
                "global_id": gid,
                "records": sum(len(v) for v in frames.values()),
                "unique_frames": len(frames),
                "first_proc_idx": indices[0] if indices else None,
                "last_proc_idx": indices[-1] if indices else None,
                "span_frames": span,
                "span_coverage": len(frames) / span if span else 0.0,
                "gap_count": sum(g > 1 for g in gaps),
                "max_gap_frames": max(gaps, default=0),
                "collision_frames": sum(len(v) > 1 for v in frames.values()),
            }
        )

    return {
        "total_records": len(records),
        "total_identities": len(grouped),
        "total_proc_frames": total_proc_frames,
        "frames_with_detections": len(per_frame),
        "mean_detections_per_frame": (
            len(records) / total_proc_frames if total_proc_frames else 0.0
        ),
        "max_detections_in_frame": max(per_frame.values(), default=0),
        "duplicate_identity_frame_keys": len(duplicate_keys),
        "duplicate_identity_frame_rows": [
            {"global_id": gid, "proc_idx": frame, "row_count": count}
            for gid, frame, count in sorted(duplicate_keys, key=lambda x: (x[1], x[0]))
        ],
        "identities": identities,
    }

