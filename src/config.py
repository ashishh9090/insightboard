"""Application configuration — reads from env vars, st.secrets, or defaults.

Design decision: we try st.secrets first (Streamlit Cloud), then os.environ
(local dev / Docker), then fall back to safe defaults. No hardcoded secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _get(key: str, default: str = "") -> str:
    """Read a config value from st.secrets → env → default."""
    # Try Streamlit secrets first (only works inside a Streamlit runtime)
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:  # noqa: BLE001 — ImportError or runtime not started
        pass
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # Database
    DB_BACKEND: str = field(default_factory=lambda: _get("DB_BACKEND", "sqlite"))
    SQLITE_PATH: Path = field(default_factory=lambda: Path(_get("SQLITE_PATH", "insightboard.db")))
    POSTGRES_DSN: str = field(default_factory=lambda: _get("POSTGRES_DSN", ""))

    # LLM (optional)
    ENABLE_LLM_SUMMARY: bool = field(
        default_factory=lambda: _get("ENABLE_LLM_SUMMARY", "false").lower() == "true"
    )
    LLM_API_KEY: str = field(default_factory=lambda: _get("LLM_API_KEY", ""))
    LLM_BASE_URL: str = field(
        default_factory=lambda: _get("LLM_BASE_URL", "https://api.openai.com/v1")
    )
    LLM_MODEL: str = field(default_factory=lambda: _get("LLM_MODEL", "gpt-4o-mini"))

    # App
    LOG_LEVEL: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO"))
    MAX_UPLOAD_MB: int = field(default_factory=lambda: int(_get("MAX_UPLOAD_MB", "50")))
    CACHE_TTL_SECONDS: int = field(default_factory=lambda: int(_get("CACHE_TTL_SECONDS", "300")))

    # Paths
    SAMPLE_DATA_DIR: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "sample"
    )


# Module-level singleton — import this everywhere
settings = Settings()
