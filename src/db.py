"""Database abstraction layer.

Exposes a ``DatabaseBackend`` protocol so the app is decoupled from the
concrete engine. The default ``SQLiteBackend`` uses SQLAlchemy Core for
portability; swapping to PostgreSQL means implementing the same protocol
with a different connection string.

Design decision: we use SQLAlchemy Core (not ORM) because we're dealing
with dynamic, user-uploaded schemas — an ORM would fight us here.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

import pandas as pd
import sqlalchemy as sa

from src.config import settings
from src.logging_setup import get_logger, metrics

logger = get_logger(__name__)


@runtime_checkable
class DatabaseBackend(Protocol):
    """Interface that any database backend must satisfy."""

    def execute_query(self, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Run a SELECT and return results as a DataFrame."""
        ...

    def insert_dataframe(self, df: pd.DataFrame, table: str, if_exists: str = "replace") -> int:
        """Write a DataFrame to a table. Returns row count inserted."""
        ...

    def list_tables(self) -> list[str]:
        """Return names of all user-data tables."""
        ...

    def table_info(self, table: str) -> pd.DataFrame:
        """Return column name/type metadata for a table."""
        ...

    def drop_table(self, table: str) -> None:
        """Drop a table if it exists."""
        ...


class SQLiteBackend:
    """SQLAlchemy-backed SQLite implementation of ``DatabaseBackend``."""

    def __init__(self, db_path: str | None = None) -> None:
        path = db_path or str(settings.SQLITE_PATH)
        self._engine = sa.create_engine(
            f"sqlite:///{path}",
            connect_args={"check_same_thread": False},
            echo=False,
        )
        logger.info("SQLite backend initialized", extra={"db_path": path})

    @property
    def engine(self) -> sa.Engine:
        return self._engine

    def execute_query(self, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Execute a read query and return a DataFrame."""
        t0 = time.perf_counter()
        try:
            with self._engine.connect() as conn:
                result = conn.execute(sa.text(query), params or {})
                df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
            elapsed = (time.perf_counter() - t0) * 1000
            metrics.record_time("query", elapsed)
            logger.debug(
                "query executed",
                extra={"duration_ms": round(elapsed, 1), "rows": len(df)},
            )
            return df
        except Exception:
            metrics.increment("errors")
            logger.exception("query failed")
            raise

    def insert_dataframe(self, df: pd.DataFrame, table: str, if_exists: str = "replace") -> int:
        """Insert a DataFrame into the database."""
        t0 = time.perf_counter()
        try:
            rows = df.to_sql(table, self._engine, if_exists=if_exists, index=False)
            elapsed = (time.perf_counter() - t0) * 1000
            metrics.record_time("insert", elapsed)
            row_count = rows if rows is not None else len(df)
            logger.info(
                "dataframe inserted",
                extra={
                    "table": table,
                    "rows": row_count,
                    "duration_ms": round(elapsed, 1),
                },
            )
            return row_count
        except Exception:
            metrics.increment("errors")
            logger.exception("insert failed", extra={"table": table})
            raise

    def list_tables(self) -> list[str]:
        """List all tables in the database."""
        inspector = sa.inspect(self._engine)
        return inspector.get_table_names()

    def table_info(self, table: str) -> pd.DataFrame:
        """Get column metadata for a table."""
        inspector = sa.inspect(self._engine)
        columns = inspector.get_columns(table)
        return pd.DataFrame([{"column": c["name"], "type": str(c["type"])} for c in columns])

    def drop_table(self, table: str) -> None:
        """Drop a table if it exists."""
        with self._engine.connect() as conn:
            conn.execute(sa.text(f'DROP TABLE IF EXISTS "{table}"'))
            conn.commit()
        logger.info("table dropped", extra={"table": table})


def get_backend(backend_type: str | None = None, **kwargs: Any) -> DatabaseBackend:
    """Factory function — returns the configured database backend.

    Args:
        backend_type: ``"sqlite"`` or ``"postgres"``. Defaults to config.
        **kwargs: Passed to the backend constructor.

    Returns:
        A ``DatabaseBackend`` instance.

    Raises:
        ValueError: If the backend type is unknown.
    """
    kind = (backend_type or settings.DB_BACKEND).lower()
    if kind == "sqlite":
        return SQLiteBackend(**kwargs)
    if kind in ("postgres", "postgresql"):
        raise NotImplementedError(
            "PostgreSQL backend is planned for v2. "
            "Implement the DatabaseBackend protocol with asyncpg or psycopg."
        )
    raise ValueError(f"Unknown DB backend: {kind!r}")
