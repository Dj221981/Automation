"""Security primitives used by automation agents."""

from .audit_logger import SecurityAuditLogger
from .auth import AuthenticatedIdentity, CapabilityACL, TokenAuthenticator
from .encryption import EncryptionManager
from .validators import sanitize_text, validate_task_payload

__all__ = [
    "AuthenticatedIdentity",
    "CapabilityACL",
    "EncryptionManager",
    "SecurityAuditLogger",
    "TokenAuthenticator",
    "sanitize_text",
    "validate_task_payload",
]
