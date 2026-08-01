from __future__ import annotations

import base64
import hashlib
from typing import Optional

from src.security.secrets import get_optional_secret, get_required_secret


class EncryptionManager:
    """Symmetric encryption helper for sensitive task and audit fields."""

    def __init__(self, secret_key: Optional[str] = None) -> None:
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise ImportError(
                "cryptography is required for EncryptionManager."
            ) from exc

        self._fernet_cls = Fernet
        resolved_key = secret_key or get_optional_secret("ENCRYPTION_KEY")

        if resolved_key is None:
            base_secret = get_required_secret("APP_SECRET_KEY")
            digest = hashlib.sha256(base_secret.encode("utf-8")).digest()
            resolved_key = base64.urlsafe_b64encode(digest).decode("utf-8")

        self._fernet = self._fernet_cls(resolved_key.encode("utf-8"))

    def encrypt(self, plaintext: str) -> str:
        if not isinstance(plaintext, str):
            raise TypeError("plaintext must be a string")
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def decrypt(self, ciphertext: str) -> str:
        if not isinstance(ciphertext, str):
            raise TypeError("ciphertext must be a string")
        return self._fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
