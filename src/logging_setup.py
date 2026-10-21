"""Structured JSON logging for InsightBoard.

All log entries are JSON objects with consistent fields:
  timestamp, level, logger, message, and optional extras (duration_ms, etc.)

Usage:
    from src.logging_setup import get_logger
    logger = get_logger(__name__)
    logger.info("query executed", extra={"duration_ms": 42, "rows": 1500})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extras the caller passed (e.g. duration_ms, rows, error)
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in logging.LogRecord(
                    "", 0, "", 0, "", (), None
                ).__dict__ and key not in ("message", "msg", "args"):
                    log_entry[key] = value
        return json.dumps(log_entry, default=str)


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Get or create a JSON-formatted logger.

    Args:
        name: Logger name, typically ``__name__``.
        level: Override log level. Defaults to config.LOG_LEVEL.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    if level is None:
        from src.config import settings

        level = settings.LOG_LEVEL
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


class AppMetrics:
    """In-memory counters for the System Metrics expander.

    Thread-safety: Streamlit runs each session in a single thread,
    so simple dict mutations are safe here. For multi-worker deployments,
    swap to ``threading.Lock`` or Redis counters.
    """

    def __init__(self) -> None:
        self._timers: dict[str, list[float]] = {}
        self._counters: dict[str, int] = {}
        self._timestamps: dict[str, str] = {}

    def record_time(self, label: str, duration_ms: float) -> None:
        """Record a timed operation in milliseconds."""
        self._timers.setdefault(label, []).append(duration_ms)

    def increment(self, label: str, n: int = 1) -> None:
        """Increment an interaction or error counter."""
        self._counters[label] = self._counters.get(label, 0) + n

    def set_timestamp(self, label: str, ts: str | None = None) -> None:
        """Record a named timestamp (e.g. data_freshness)."""
        self._timestamps[label] = ts or datetime.now(UTC).isoformat()

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of all metrics."""
        timer_summary = {}
        for label, values in self._timers.items():
            timer_summary[label] = {
                "count": len(values),
                "avg_ms": round(sum(values) / len(values), 1) if values else 0,
                "max_ms": round(max(values), 1) if values else 0,
                "last_ms": round(values[-1], 1) if values else 0,
            }
        return {
            "timers": timer_summary,
            "counters": dict(self._counters),
            "timestamps": dict(self._timestamps),
        }


# Module-level singleton
metrics = AppMetrics()
