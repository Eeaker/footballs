from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class ModeObservation:
    frame: int
    mode: int | None
    confidence: float


@dataclass(frozen=True)
class ModeTransition:
    frame: int
    before_mode: int
    after_mode: int
    before_purity: float
    after_purity: float
    before_evidence: int
    after_evidence: int


def _majority(rows: list[ModeObservation]) -> tuple[int | None, float, int]:
    counts = Counter(row.mode for row in rows if row.mode is not None)
    if not counts:
        return None, 0.0, 0
    mode, count = counts.most_common(1)[0]
    total = sum(counts.values())
    return int(mode), count / total, total


def detect_persistent_transitions(
    observations: list[ModeObservation], *, window: int = 8,
    minimum_evidence: int = 5, purity: float = .75,
    minimum_confidence: float = .35, span_frames: int | None = None,
    cooldown_frames: int = 12, reversal_horizon_frames: int = 30,
) -> list[ModeTransition]:
    """Find sustained self-mode changes while ignoring uncertain frames and flicker.

    No semantic colour name or manually supplied team prototype is used.  A cut
    requires a high-purity mode on both sides of a boundary.  The reported cut
    frame is the first trusted observation of the new mode.
    """
    ordered = sorted(observations, key=lambda row: row.frame)
    if len(ordered) < minimum_evidence * 2:
        return []
    span = span_frames if span_frames is not None else max(window * 3, 18)
    trusted = [row for row in ordered if row.mode is not None and row.confidence >= minimum_confidence]
    if len(trusted) < minimum_evidence * 2:
        return []

    candidates: list[ModeTransition] = []
    for index in range(1, len(ordered)):
        boundary = ordered[index].frame
        left = [row for row in ordered[:index]
                if boundary - span <= row.frame < boundary
                and row.mode is not None and row.confidence >= minimum_confidence][-window:]
        right = [row for row in ordered[index:]
                 if boundary <= row.frame <= boundary + span
                 and row.mode is not None and row.confidence >= minimum_confidence][:window]
        before, before_purity, before_count = _majority(left)
        after, after_purity, after_count = _majority(right)
        if before is None or after is None or before == after:
            continue
        if before_count < minimum_evidence or after_count < minimum_evidence:
            continue
        if before_purity < purity or after_purity < purity:
            continue
        new_rows = [row for row in right if row.mode == after]
        if not new_rows:
            continue
        candidates.append(ModeTransition(
            frame=new_rows[0].frame, before_mode=before, after_mode=after,
            before_purity=before_purity, after_purity=after_purity,
            before_evidence=before_count, after_evidence=after_count,
        ))

    accepted: list[ModeTransition] = []
    for candidate in candidates:
        if accepted and candidate.frame - accepted[-1].frame < cooldown_frames:
            continue
        if accepted and candidate.before_mode == accepted[-1].before_mode \
                and candidate.after_mode == accepted[-1].after_mode:
            continue
        accepted.append(candidate)
    filtered: list[ModeTransition] = []
    index = 0
    while index < len(accepted):
        current = accepted[index]
        if index + 1 < len(accepted):
            following = accepted[index + 1]
            is_short_reversal = (
                following.frame - current.frame <= reversal_horizon_frames
                and current.before_mode == following.after_mode
                and current.after_mode == following.before_mode
            )
            if is_short_reversal:
                index += 2
                continue
        filtered.append(current)
        index += 1
    return filtered
