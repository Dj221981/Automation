from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from threading import RLock
from typing import Any, Dict, List, Optional


@dataclass
class StoredTask:
    id: str
    description: str
    priority: str
    assigned_to: Optional[str] = None
    status: str = "PENDING"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskStore(ABC):
    @abstractmethod
    def create_task(self, task: StoredTask) -> None: ...

    @abstractmethod
    def update_task(self, task: StoredTask) -> None: ...

    @abstractmethod
    def get_task(self, task_id: str) -> Optional[StoredTask]: ...

    @abstractmethod
    def list_tasks(self, status: Optional[str] = None) -> List[StoredTask]: ...


class InMemoryTaskStore(TaskStore):
    def __init__(self) -> None:
        self._tasks: Dict[str, StoredTask] = {}
        self._lock = RLock()

    def create_task(self, task: StoredTask) -> None:
        with self._lock:
            if task.id in self._tasks:
                raise ValueError(f"Task already exists: {task.id}")
            self._tasks[task.id] = task

    def update_task(self, task: StoredTask) -> None:
        with self._lock:
            if task.id not in self._tasks:
                raise KeyError(f"Task not found: {task.id}")
            self._tasks[task.id] = task

    def get_task(self, task_id: str) -> Optional[StoredTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self, status: Optional[str] = None) -> List[StoredTask]:
        with self._lock:
            values = list(self._tasks.values())
            if status is None:
                return values
            return [t for t in values if t.status == status]


class RedisTaskStore(TaskStore):
    """
    Redis-backed task store.

    Key schema:
      - task hash:         agent:task:{task_id}
      - status index set:  agent:tasks:status:{STATUS}
      - all tasks set:     agent:tasks:all

    Serialization:
      - `parameters`, `dependencies`, `metadata`, `result` are JSON strings.
      - datetimes are ISO-8601 strings.
    """

    def __init__(self, redis_url: str, key_prefix: str = "agent") -> None:
        try:
            import redis  # type: ignore
        except ImportError as exc:
            raise ImportError("redis package is required for RedisTaskStore. Install `redis>=5`.") from exc

        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._prefix = key_prefix

    def _task_key(self, task_id: str) -> str:
        return f"{self._prefix}:task:{task_id}"

    def _status_key(self, status: str) -> str:
        return f"{self._prefix}:tasks:status:{status}"

    def _all_key(self) -> str:
        return f"{self._prefix}:tasks:all"

    @staticmethod
    def _dt_to_str(value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None

    @staticmethod
    def _str_to_dt(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        return datetime.fromisoformat(value)

    @staticmethod
    def _task_to_hash(task: StoredTask) -> Dict[str, str]:
        payload = asdict(task)
        payload["created_at"] = task.created_at.isoformat()
        payload["completed_at"] = task.completed_at.isoformat() if task.completed_at else ""
        payload["parameters"] = json.dumps(task.parameters)
        payload["dependencies"] = json.dumps(task.dependencies)
        payload["metadata"] = json.dumps(task.metadata)
        payload["result"] = json.dumps(task.result)
        payload["assigned_to"] = task.assigned_to or ""
        payload["error"] = task.error or ""
        return {k: str(v) for k, v in payload.items()}

    @staticmethod
    def _hash_to_task(data: Dict[str, str]) -> StoredTask:
        return StoredTask(
            id=data["id"],
            description=data.get("description", ""),
            priority=data.get("priority", "NORMAL"),
            assigned_to=data.get("assigned_to") or None,
            status=data.get("status", "PENDING"),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            result=json.loads(data.get("result", "null")),
            error=data.get("error") or None,
            parameters=json.loads(data.get("parameters", "{}")),
            dependencies=json.loads(data.get("dependencies", "[]")),
            metadata=json.loads(data.get("metadata", "{}")),
        )

    def create_task(self, task: StoredTask) -> None:
        key = self._task_key(task.id)
        status_key = self._status_key(task.status)
        all_key = self._all_key()

        task_hash = self._task_to_hash(task)

        with self._redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    pipe.watch(key)
                    if pipe.exists(key):
                        raise ValueError(f"Task already exists: {task.id}")
                    pipe.multi()
                    pipe.hset(key, mapping=task_hash)
                    pipe.sadd(status_key, task.id)
                    pipe.sadd(all_key, task.id)
                    pipe.execute()
                    break
                except Exception:
                    pipe.reset()
                    raise

    def update_task(self, task: StoredTask) -> None:
        key = self._task_key(task.id)

        with self._redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    pipe.watch(key)
                    existing = pipe.hgetall(key)
                    if not existing:
                        raise KeyError(f"Task not found: {task.id}")

                    old_status = existing.get("status", "PENDING")
                    new_status = task.status

                    pipe.multi()
                    pipe.hset(key, mapping=self._task_to_hash(task))
                    if old_status != new_status:
                        pipe.srem(self._status_key(old_status), task.id)
                        pipe.sadd(self._status_key(new_status), task.id)
                    pipe.sadd(self._all_key(), task.id)
                    pipe.execute()
                    break
                except Exception:
                    pipe.reset()
                    raise

    def get_task(self, task_id: str) -> Optional[StoredTask]:
        data = self._redis.hgetall(self._task_key(task_id))
        if not data:
            return None
        return self._hash_to_task(data)

    def list_tasks(self, status: Optional[str] = None) -> List[StoredTask]:
        if status:
            ids = self._redis.smembers(self._status_key(status))
        else:
            ids = self._redis.smembers(self._all_key())

        tasks: List[StoredTask] = []
        if not ids:
            return tasks

        with self._redis.pipeline(transaction=False) as pipe:
            for task_id in ids:
                pipe.hgetall(self._task_key(task_id))
            rows = pipe.execute()

        for row in rows:
            if row:
                tasks.append(self._hash_to_task(row))

        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks
