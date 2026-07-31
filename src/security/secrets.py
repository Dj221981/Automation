from __future__ import annotations

import os


def get_required_secret(name: str) -> str:
    """Load a required secret from environment variables."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment secret: {name}")
    return value


def get_optional_secret(name: str, default: str | None = None) -> str | None:
    """Load an optional secret from environment variables."""
    return os.getenv(name, default)
