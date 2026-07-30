from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)


class SecurityAuditLogger:
    """Structured security audit logger with safe payload redaction."""

    REDACT_KEYS = {"token", "secret", "password", "authorization"}

    def __init__(self, logger_name: str = "security_audit") -> None:
        self._logger = logging.getLogger(logger_name)

    def _redact(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        redacted: Dict[str, Any] = {}
        for key, value in payload.items():
            if key.lower() in self.REDACT_KEYS:
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = value
        return redacted

    def log_event(self, event_type: str, actor: str, details: Dict[str, Any]) -> None:
        if not event_type:
            raise ValueError("event_type is required")
        if not actor:
            raise ValueError("actor is required")
        if not isinstance(details, dict):
            raise TypeError("details must be a dictionary")

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "actor": actor,
            "details": self._redact(details),
        }
        self._logger.info(json.dumps(payload, sort_keys=True))
        logger.debug("Security audit event logged: %s", event_type)
