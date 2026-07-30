from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from src.agents.task_store import (StoredTask, TaskStore,
                                   ensure_valid_transition,
                                   normalize_task_status)

logger = logging.getLogger(__name__)


class PostgresTaskStore(TaskStore):
    """PostgreSQL-backed task store with strict status transition validation."""

    def __init__(self, dsn: str, table_name: str = "tasks") -> None:
        if not dsn or not isinstance(dsn, str):
            raise ValueError("dsn must be a non-empty string")
        if not table_name or not isinstance(table_name, str):
            raise ValueError("table_name must be a non-empty string")

        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise ImportError(
                "psycopg is required for PostgresTaskStore. Install `psycopg[binary]>=3`."
            ) from exc

        self._psycopg = psycopg
        self._dict_row = dict_row
        self._dsn = dsn
        self._table_name = table_name

    def _connect(self):
        return self._psycopg.connect(self._dsn, row_factory=self._dict_row)

    def _to_payload(self, task: StoredTask) -> Dict[str, Any]:
        payload = asdict(task)
        payload["created_at"] = task.created_at
        payload["completed_at"] = task.completed_at
        payload["parameters"] = json.dumps(task.parameters)
        payload["dependencies"] = json.dumps(task.dependencies)
        payload["metadata"] = json.dumps(task.metadata)
        payload["result"] = json.dumps(task.result)
        return payload

    @staticmethod
    def _from_row(row: Dict[str, Any]) -> StoredTask:
        return StoredTask(
            id=row["id"],
            description=row["description"],
            priority=row["priority"],
            assigned_to=row.get("assigned_to"),
            status=row.get("status", "PENDING"),
            created_at=row["created_at"],
            completed_at=row.get("completed_at"),
            result=json.loads(row["result"]) if row.get("result") is not None else None,
            error=row.get("error"),
            parameters=json.loads(row.get("parameters") or "{}"),
            dependencies=json.loads(row.get("dependencies") or "[]"),
            metadata=json.loads(row.get("metadata") or "{}"),
        )

    def create_task(self, task: StoredTask) -> None:
        payload = self._to_payload(task)

        sql = f"""
        INSERT INTO {self._table_name}
            (
                id,
                description,
                priority,
                assigned_to,
                status,
                created_at,
                completed_at,
                result,
                error,
                parameters,
                dependencies,
                metadata
            )
        VALUES
            (
                %(id)s,
                %(description)s,
                %(priority)s,
                %(assigned_to)s,
                %(status)s,
                %(created_at)s,
                %(completed_at)s,
                %(result)s,
                %(error)s,
                %(parameters)s::jsonb,
                %(dependencies)s::jsonb,
                %(metadata)s::jsonb
            )
        """

        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(sql, payload)
                conn.commit()
        except Exception as exc:
            logger.exception("Failed to create task %s", task.id)
            raise RuntimeError(f"failed to create task {task.id}: {exc}") from exc

    def update_task(self, task: StoredTask) -> None:
        payload = self._to_payload(task)

        select_sql = f"SELECT status FROM {self._table_name} WHERE id = %s"
        update_sql = f"""
        UPDATE {self._table_name}
        SET description = %(description)s,
            priority = %(priority)s,
            assigned_to = %(assigned_to)s,
            status = %(status)s,
            created_at = %(created_at)s,
            completed_at = %(completed_at)s,
            result = %(result)s::jsonb,
            error = %(error)s,
            parameters = %(parameters)s::jsonb,
            dependencies = %(dependencies)s::jsonb,
            metadata = %(metadata)s::jsonb,
            updated_at = NOW()
        WHERE id = %(id)s
        """

        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(select_sql, (task.id,))
                existing = cur.fetchone()
                if not existing:
                    raise KeyError(f"Task not found: {task.id}")

                ensure_valid_transition(existing["status"], task.status)
                cur.execute(update_sql, payload)
                conn.commit()
        except KeyError:
            raise
        except Exception as exc:
            logger.exception("Failed to update task %s", task.id)
            raise RuntimeError(f"failed to update task {task.id}: {exc}") from exc

    def get_task(self, task_id: str) -> Optional[StoredTask]:
        sql = f"SELECT * FROM {self._table_name} WHERE id = %s"

        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(sql, (task_id,))
                row = cur.fetchone()
                return self._from_row(row) if row else None
        except Exception as exc:
            logger.exception("Failed to fetch task %s", task_id)
            raise RuntimeError(f"failed to fetch task {task_id}: {exc}") from exc

    def list_tasks(self, status: Optional[str] = None) -> List[StoredTask]:
        params: tuple[Any, ...] = tuple()
        sql = f"SELECT * FROM {self._table_name}"

        if status:
            normalized = normalize_task_status(status)
            sql += " WHERE status = %s"
            params = (normalized,)

        sql += " ORDER BY created_at DESC"

        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                return [self._from_row(row) for row in rows]
        except Exception as exc:
            logger.exception("Failed to list tasks")
            raise RuntimeError(f"failed to list tasks: {exc}") from exc
