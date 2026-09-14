from __future__ import annotations

import json
import os
from typing import Any


def database_enabled() -> bool:
    return os.getenv("FOOTBALL_INSIGHT_STORAGE_MODE", "file").strip().lower() == "database"


def connect():
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("数据库模式已启用，但未配置 DATABASE_URL")
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("缺少 PostgreSQL 驱动，请使用一键容器部署") from exc
    return psycopg.connect(url)


class PostgresProjectRepository:
    def save(self, project: dict[str, Any]) -> dict[str, Any]:
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO core.projects (id, kind, status, payload, created_at, updated_at)
                VALUES (%s, %s, %s, %s::jsonb, %s::timestamptz, %s::timestamptz)
                ON CONFLICT (id) DO UPDATE SET
                  kind=EXCLUDED.kind, status=EXCLUDED.status, payload=EXCLUDED.payload,
                  updated_at=EXCLUDED.updated_at
                """,
                (project["id"], project.get("kind", "analysis"), project.get("status", "draft"),
                 json.dumps(project, ensure_ascii=False), project["created_at"], project["updated_at"]),
            )
        return project

    def load(self, project_id: str) -> dict[str, Any]:
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT payload FROM core.projects WHERE id=%s", (project_id,))
            row = cursor.fetchone()
        if row is None:
            raise FileNotFoundError(project_id)
        return row[0] if isinstance(row[0], dict) else json.loads(row[0])

    def list(self) -> list[dict[str, Any]]:
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT payload FROM core.projects ORDER BY (kind='demo'), updated_at DESC")
            rows = cursor.fetchall()
        return [row[0] if isinstance(row[0], dict) else json.loads(row[0]) for row in rows]

    def delete(self, project_id: str) -> None:
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM core.projects WHERE id=%s", (project_id,))


repository = PostgresProjectRepository()
