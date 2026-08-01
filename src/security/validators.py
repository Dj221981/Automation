from __future__ import annotations

import html
import re
from typing import Any, Dict

_TASK_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def sanitize_text(value: str, *, max_length: int = 4096) -> str:
    """Trim, bound, and HTML-escape untrusted text input."""
    if not isinstance(value, str):
        raise TypeError("value must be a string")

    sanitized = html.escape(value.strip())
    if not sanitized:
        raise ValueError("value cannot be empty")
    if len(sanitized) > max_length:
        raise ValueError(f"value length exceeds max_length={max_length}")
    return sanitized


def validate_task_id(task_id: str) -> str:
    """Validate task IDs to avoid malformed identifiers and key injection."""
    if not isinstance(task_id, str):
        raise TypeError("task_id must be a string")
    if not _TASK_ID_RE.match(task_id):
        raise ValueError("task_id must match ^[A-Za-z0-9_-]{1,128}$")
    return task_id


def validate_task_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize a task payload dictionary."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary")

    required_keys = {"id", "description", "priority"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(f"payload missing required fields: {sorted(missing)}")

    validated = dict(payload)
    validated["id"] = validate_task_id(str(payload["id"]))
    validated["description"] = sanitize_text(
        str(payload["description"]), max_length=8192
    )
    validated["priority"] = sanitize_text(
        str(payload["priority"]), max_length=32
    ).upper()

    metadata = payload.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary when provided")

    validated_metadata: Dict[str, Any] = {}
    for key, value in metadata.items():
        validated_key = sanitize_text(str(key), max_length=128)
        if isinstance(value, str):
            validated_metadata[validated_key] = sanitize_text(value, max_length=4096)
        else:
            validated_metadata[validated_key] = value
    validated["metadata"] = validated_metadata
    return validated
