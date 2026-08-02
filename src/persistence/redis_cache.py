from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RedisCache:
    """JSON-friendly Redis cache wrapper with optional TTL support."""

    def __init__(
        self,
        redis_url: str,
        key_prefix: str = "automation",
        default_ttl_seconds: int = 300,
    ):
        if not redis_url:
            raise ValueError("redis_url must be a non-empty string")
        if default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be greater than 0")

        try:
            import redis  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "redis package is required for RedisCache. Install `redis>=5`."
            ) from exc

        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._key_prefix = key_prefix
        self._default_ttl_seconds = default_ttl_seconds

    def _key(self, key: str) -> str:
        return f"{self._key_prefix}:{key}"

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        if not key:
            raise ValueError("key must be a non-empty string")

        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        if ttl <= 0:
            raise ValueError("ttl_seconds must be greater than 0")

        payload = json.dumps(value)
        self._redis.setex(self._key(key), ttl, payload)

    def get(self, key: str) -> Optional[Any]:
        if not key:
            raise ValueError("key must be a non-empty string")

        payload = self._redis.get(self._key(key))
        if payload is None:
            return None

        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Non-JSON payload found in Redis key %s", key)
            return payload

    def delete(self, key: str) -> bool:
        if not key:
            raise ValueError("key must be a non-empty string")
        return bool(self._redis.delete(self._key(key)))

    def exists(self, key: str) -> bool:
        if not key:
            raise ValueError("key must be a non-empty string")
        return bool(self._redis.exists(self._key(key)))
