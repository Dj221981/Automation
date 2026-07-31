from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set

from src.security.secrets import get_required_secret


@dataclass(frozen=True)
class AuthenticatedIdentity:
    subject: str
    role: str
    capabilities: Set[str]


class CapabilityACL:
    """Simple role-to-capability ACL for agent operations."""

    def __init__(self, role_capabilities: Optional[Dict[str, Iterable[str]]] = None):
        defaults = {
            "admin": {"*"},
            "orchestrator": {"task:create", "task:list", "task:assign"},
            "executor": {"task:execute", "task:list"},
            "viewer": {"task:list"},
        }
        source = role_capabilities or defaults
        self._capabilities = {role: set(values) for role, values in source.items()}

    def is_allowed(self, role: str, capability: str) -> bool:
        if role not in self._capabilities:
            return False
        capabilities = self._capabilities[role]
        return "*" in capabilities or capability in capabilities


class TokenAuthenticator:
    """HMAC token auth with env-managed secrets and capability checks."""

    def __init__(self, secret: Optional[str] = None):
        self._secret = (secret or get_required_secret("AGENT_AUTH_SECRET")).encode(
            "utf-8"
        )

    def issue_token(
        self,
        subject: str,
        role: str,
        capabilities: Iterable[str],
        ttl_seconds: int = 3600,
    ) -> str:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than 0")

        payload = {
            "sub": subject,
            "role": role,
            "capabilities": sorted(set(capabilities)),
            "exp": int(time.time()) + ttl_seconds,
        }
        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode(
            "utf-8"
        )

        signature = hmac.new(
            self._secret, payload_b64.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return f"{payload_b64}.{signature}"

    def verify_token(self, token: str) -> AuthenticatedIdentity:
        if not token or "." not in token:
            raise ValueError("Invalid token format")

        payload_b64, signature = token.rsplit(".", 1)
        expected = hmac.new(
            self._secret, payload_b64.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("Invalid token signature")

        payload_json = base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode(
            "utf-8"
        )
        payload = json.loads(payload_json)
        if int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("Token expired")

        return AuthenticatedIdentity(
            subject=str(payload["sub"]),
            role=str(payload["role"]),
            capabilities=set(payload.get("capabilities", [])),
        )
