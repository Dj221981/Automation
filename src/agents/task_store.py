from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from threading import RLock
from typing import Any, Dict, List, Optional


ALLOWED_TASK_STATUSES = {
    "PENDING",
    "ASSIGNED",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
}

ALLOWED_TASK_TRANSITIONS = {
    "PENDING": {"ASSIGNED", "CANCELLED"},
    "ASSIGNED": {"RUNNING", "PENDING", "CANCELLED"},
    "RUNNING": {"COMPLETED", "FAILED", "CANCELLED", "PENDING"},
    "COMPLETED": set(),
    "FAILED": {"PENDING"},  # Allow explicit requeue / recovery
    "CANCELLED": set(),
}


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

    def __post_init__(self) -> None:
        self.description = self.description.strip()
        if not self.id.strip():
            raise ValueError("Task id cannot be empty")
        if not self.description:
            raise ValueError("Task description cannot be empty")
        self.status = normalize_task_status(self.status)
        if self.completed_at and self.completed_at < self.created_at:
            raise ValueError("completed_at cannot be earlier than created_at")


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
        task = _validated_copy(task)
        with self._lock:
            if task.id in self._tasks:
                raise ValueError(f"Task already exists: {task.id}")
            self._tasks[task.id] = task

    def update_task(self, task: StoredTask) -> None:
        task = _validated_copy(task)
        with self._lock:
            existing = self._tasks.get(task.id)
            if existing is None:
                raise KeyError(f"Task not found: {task.id}")
            ensure_valid_transition(existing.status, task.status)
            self._tasks[task.id] = task

    def get_task(self, task_id: str) -> Optional[StoredTask]:
        with self._lock:
            task = self._tasks.get(task_id)
            return _validated_copy(task) if task else None

    def list_tasks(self, status: Optional[str] = None) -> List[StoredTask]:
        normalized_status = normalize_task_status(status) if status is not None else None
        with self._lock:
            values = [_validated_copy(task) for task in self._tasks.values()]
            if normalized_status is None:
                return sorted(values, key=lambda t: t.created_at, reverse=True)
            return sorted([t for t in values if t.status == normalized_status], key=lambda t: t.created_at, reverse=True)


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
        return f"{self._prefix}:tasks:status:{normalize_task_status(status)}"

    def _all_key(self) -> str:
        return f"{self._prefix}:tasks:all"

    @staticmethod
    def _task_to_hash(task: StoredTask) -> Dict[str, str]:
        safe_task = _validated_copy(task)
        payload = asdict(safe_task)
        payload["created_at"] = safe_task.created_at.isoformat()
        payload["completed_at"] = safe_task.completed_at.isoformat() if safe_task.completed_at else ""
        payload["parameters"] = json.dumps(safe_task.parameters)
        payload["dependencies"] = json.dumps(safe_task.dependencies)
        payload["metadata"] = json.dumps(safe_task.metadata)
        payload["result"] = json.dumps(safe_task.result)
        payload["assigned_to"] = safe_task.assigned_to or ""
        payload["error"] = safe_task.error or ""
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
        validated_task = _validated_copy(task)
        key = self._task_key(validated_task.id)
        status_key = self._status_key(validated_task.status)
        all_key = self._all_key()

        task_hash = self._task_to_hash(validated_task)

        with self._redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    pipe.watch(key)
                    if pipe.exists(key):
                        raise ValueError(f"Task already exists: {validated_task.id}")
                    pipe.multi()
                    pipe.hset(key, mapping=task_hash)
                    pipe.sadd(status_key, validated_task.id)
                    pipe.sadd(all_key, validated_task.id)
                    pipe.execute()
                    break
                except Exception:
                    pipe.reset()
                    raise

    def update_task(self, task: StoredTask) -> None:
        validated_task = _validated_copy(task)
        key = self._task_key(validated_task.id)

        with self._redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    pipe.watch(key)
                    existing = pipe.hgetall(key)
                    if not existing:
                        raise KeyError(f"Task not found: {validated_task.id}")

                    old_status = normalize_task_status(existing.get("status", "PENDING"))
                    new_status = normalize_task_status(validated_task.status)
                    ensure_valid_transition(old_status, new_status)

                    pipe.multi()
                    pipe.hset(key, mapping=self._task_to_hash(validated_task))
                    if old_status != new_status:
                        pipe.srem(self._status_key(old_status), validated_task.id)
                        pipe.sadd(self._status_key(new_status), validated_task.id)
                    pipe.sadd(self._all_key(), validated_task.id)
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
        normalized_status = normalize_task_status(status) if status else None
        if normalized_status:
            ids = self._redis.smembers(self._status_key(normalized_status))
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


def normalize_task_status(status: str) -> str:
    normalized = status.strip().upper()
    if normalized not in ALLOWED_TASK_STATUSES:
        raise ValueError(f"Invalid task status: {status}")
    return normalized


def ensure_valid_transition(current_status: str, new_status: str) -> None:
    current = normalize_task_status(current_status)
    new = normalize_task_status(new_status)
    if current == new:
        return
    if new not in ALLOWED_TASK_TRANSITIONS[current]:
        raise ValueError(f"Invalid task transition: {current} -> {new}")


def _validated_copy(task: StoredTask) -> StoredTask:
    if task is None:
        raise ValueError("Task cannot be None")
    return StoredTask(
        id=task.id,
        description=task.description,
        priority=task.priority,
        assigned_to=task.assigned_to,
        status=task.status,
        created_at=task.created_at,
        completed_at=task.completed_at,
        result=task.result,
        error=task.error,
        parameters=dict(task.parameters),
        dependencies=list(task.dependencies),
        metadata=dict(task.metadata),
    )
