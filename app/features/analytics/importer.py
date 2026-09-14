from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.features.projects.repository import connect, database_enabled


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def persist_analysis_tables(project: dict[str, Any], outputs: Path) -> None:
    """Load durable, queryable results into partitioned PostgreSQL tables."""
    if not database_enabled():
        return
    project_id = project["id"]
    mot = outputs / "tracking" / "tracking" / "tracking_mot.txt"
    events_path = outputs / "events_for_annotation.json"
    catalog_path = outputs / "identity_resolution" / "filtered" / "identity_catalog.json"
    team_rows = {int(float(row["global_id"])): row.get("team_id") for row in _read_csv(outputs / "match_analysis" / "analysis" / "player_team_map.csv") if row.get("global_id")}
    ocr_rows = {int(float(row["global_id"])): row for row in _read_csv(outputs / "number_ocr" / "jersey_number_results.csv") if row.get("global_id")}

    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("DELETE FROM tracking.observations WHERE project_id=%s", (project_id,))
        if mot.is_file():
            with cursor.copy("COPY tracking.observations (project_id,frame_index,global_id,bbox,confidence) FROM STDIN") as copy:
                with mot.open(encoding="utf-8-sig") as handle:
                    for row in csv.reader(handle):
                        if len(row) >= 6:
                            copy.write_row((project_id, int(float(row[0])), int(float(row[1])),
                                            [float(value) for value in row[2:6]], float(row[6]) if len(row) > 6 else None))

        cursor.execute("DELETE FROM analytics.events WHERE project_id=%s", (project_id,))
        if events_path.is_file():
            payload = json.loads(events_path.read_text(encoding="utf-8-sig"))
            events = payload if isinstance(payload, list) else payload.get("events", [])
            for index, event in enumerate(events):
                event_id = str(event.get("event_id") or event.get("id") or f"event-{index}")
                frame = int(event.get("frame_index") or event.get("frame") or event.get("start_frame") or 0)
                event_type = str(event.get("event_type") or event.get("type") or "unknown")
                cursor.execute(
                    "INSERT INTO analytics.events(project_id,event_id,frame_index,event_type,payload) VALUES(%s,%s,%s,%s,%s::jsonb)",
                    (project_id, event_id, frame, event_type, json.dumps(event, ensure_ascii=False)),
                )

        cursor.execute("DELETE FROM identity.fragments WHERE project_id=%s", (project_id,))
        cursor.execute("DELETE FROM identity.players WHERE project_id=%s", (project_id,))
        if catalog_path.is_file():
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            published_sources: set[int] = set()
            for player in catalog.get("published_players", []):
                canonical = int(player["canonical_id"])
                sources = sorted({int(value) for value in player.get("source_ids", [])})
                published_sources.update(sources)
                ocr = next((ocr_rows.get(source, {}) for source in sources if "confirm" in str(ocr_rows.get(source, {}).get("status", "")).lower()), {})
                jersey = str(ocr.get("predicted_number") or "").strip() or None
                team_id = team_rows.get(canonical) or next((team_rows.get(source) for source in sources if team_rows.get(source)), None)
                label = str(ocr.get("team_label") or ocr.get("team") or "").strip() or (((project.get("match") or {}).get("team_labels") or {}).get(team_id, team_id) if team_id else None)
                display_id = f"{label}{jersey}号" if label and jersey else None
                cursor.execute(
                    "INSERT INTO identity.players(project_id,canonical_id,team_id,jersey_number,display_id,source_ids) VALUES(%s,%s,%s,%s,%s,%s)",
                    (project_id, canonical, team_id, jersey, display_id, sources),
                )
                for source in sources:
                    cursor.execute(
                        "INSERT INTO identity.fragments(project_id,source_id,canonical_id,state,evidence) VALUES(%s,%s,%s,'confirmed_merged',%s::jsonb)",
                        (project_id, source, canonical, json.dumps({"policy": catalog.get("policy")}, ensure_ascii=False)),
                    )
            for source in catalog.get("isolated_ids", []):
                source = int(source)
                if source not in published_sources:
                    cursor.execute("INSERT INTO identity.fragments(project_id,source_id,state) VALUES(%s,%s,'isolated') ON CONFLICT DO NOTHING", (project_id, source))
            for source in catalog.get("quarantine_ids", []):
                source = int(source)
                cursor.execute(
                    "INSERT INTO identity.fragments(project_id,source_id,state) VALUES(%s,%s,'quarantine') ON CONFLICT(project_id,source_id) DO UPDATE SET state='quarantine',canonical_id=NULL",
                    (project_id, source),
                )
