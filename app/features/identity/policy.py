from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class IdentityCatalog:
    """The boundary between technical trajectories and publishable players.

    A trajectory is never a player merely because it survived tracking.  Only a
    confirmed component containing at least two source IDs is publishable.  All
    other IDs remain queryable for audit, but are isolated from player results.
    """

    published: dict[int, tuple[int, ...]]
    isolated_ids: frozenset[int]
    quarantine_ids: frozenset[int]

    def source_ids(self, canonical_id: int) -> tuple[int, ...]:
        return self.published.get(int(canonical_id), ())

    def is_publishable(self, canonical_id: int) -> bool:
        return int(canonical_id) in self.published


EMPTY_CATALOG = IdentityCatalog({}, frozenset(), frozenset())


def _ints(values: Iterable[Any]) -> tuple[int, ...]:
    parsed: set[int] = set()
    for value in values:
        try:
            parsed.add(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(parsed))


def load_identity_catalog(outputs: Path) -> IdentityCatalog:
    candidates = (
        outputs / "identity_resolution" / "filtered" / "identity_catalog.json",
        outputs / "identity_catalog.json",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return EMPTY_CATALOG
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        published: dict[int, tuple[int, ...]] = {}
        for row in payload.get("published_players", []):
            canonical = int(row["canonical_id"])
            sources = _ints(row.get("source_ids", []))
            if len(sources) >= 2 and bool(row.get("confirmed", True)):
                published[canonical] = sources
        return IdentityCatalog(
            published=published,
            isolated_ids=frozenset(_ints(payload.get("isolated_ids", []))),
            quarantine_ids=frozenset(_ints(payload.get("quarantine_ids", []))),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        # Invalid catalog data fails closed: no technical IDs leak into players.
        return EMPTY_CATALOG


def confirmed_jersey(value: Any, status: Any = "") -> str | None:
    number = str(value or "").strip()
    state = str(status or "").strip().casefold()
    if not number or number in {"—", "待确认", "unknown", "none"}:
        return None
    if status and not any(token in state for token in ("confirm", "verified", "human")):
        return None
    return number


def build_display_id(team_label: Any, jersey_number: Any, fallback: str = "球员") -> str:
    """Use the OCR-confirmed semantic identity, e.g. ``蓝队25号``."""
    team = str(team_label or "").strip()
    number = str(jersey_number or "").strip()
    if team and number and number not in {"—", "待确认"}:
        return f"{team}{number}号"
    return fallback


def human_confirmed_groups(mappings: dict[str, dict[str, Any]]) -> dict[int, tuple[int, ...]]:
    """Return explicit multi-ID operator confirmations keyed by representative."""
    groups: dict[str, set[int]] = {}
    for raw_gid, mapping in mappings.items():
        try:
            gid = int(raw_gid)
        except (TypeError, ValueError):
            continue
        key = str(mapping.get("person_key") or "").strip()
        linked = set(_ints(mapping.get("linked_global_ids", [])))
        linked.add(gid)
        if key:
            groups.setdefault(key, set()).update(linked)
    return {min(ids): tuple(sorted(ids)) for ids in groups.values() if len(ids) >= 2}

