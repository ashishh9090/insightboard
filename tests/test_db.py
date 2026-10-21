"""Tests for the database abstraction layer."""

from __future__ import annotations

import pandas as pd
import pytest

from src.db import SQLiteBackend, get_backend


class TestSQLiteBackend:
    """Test SQLite backend operations."""

    def test_insert_and_query(self, tmp_db: SQLiteBackend) -> None:
        """Round-trip: insert a DataFrame, query it back."""
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        rows = tmp_db.insert_dataframe(df, "test_table")
        assert rows == 3

        result = tmp_db.execute_query("SELECT * FROM test_table")
        assert len(result) == 3
        assert list(result.columns) == ["id", "name"]

    def test_list_tables(self, tmp_db: SQLiteBackend) -> None:
        """Verify table listing."""
        df = pd.DataFrame({"x": [1]})
        tmp_db.insert_dataframe(df, "alpha")
        tmp_db.insert_dataframe(df, "beta")

        tables = tmp_db.list_tables()
        assert "alpha" in tables
        assert "beta" in tables

    def test_table_info(self, tmp_db: SQLiteBackend) -> None:
        """Verify column metadata retrieval."""
        df = pd.DataFrame({"col_a": [1], "col_b": ["text"]})
        tmp_db.insert_dataframe(df, "info_test")

        info = tmp_db.table_info("info_test")
        assert len(info) == 2
        assert "col_a" in info["column"].values
        assert "col_b" in info["column"].values

    def test_drop_table(self, tmp_db: SQLiteBackend) -> None:
        """Verify table drop."""
        df = pd.DataFrame({"x": [1]})
        tmp_db.insert_dataframe(df, "to_drop")
        assert "to_drop" in tmp_db.list_tables()

        tmp_db.drop_table("to_drop")
        assert "to_drop" not in tmp_db.list_tables()

    def test_replace_existing_table(self, tmp_db: SQLiteBackend) -> None:
        """Verify that re-inserting replaces the table."""
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [10, 20, 30]})

        tmp_db.insert_dataframe(df1, "replace_test")
        result1 = tmp_db.execute_query("SELECT COUNT(*) as cnt FROM replace_test")
        assert result1["cnt"].iloc[0] == 2

        tmp_db.insert_dataframe(df2, "replace_test", if_exists="replace")
        result2 = tmp_db.execute_query("SELECT COUNT(*) as cnt FROM replace_test")
        assert result2["cnt"].iloc[0] == 3

    def test_parameterized_query(self, tmp_db: SQLiteBackend) -> None:
        """Verify parameterized queries work."""
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        tmp_db.insert_dataframe(df, "param_test")

        result = tmp_db.execute_query(
            "SELECT * FROM param_test WHERE id = :id",
            params={"id": 2},
        )
        assert len(result) == 1
        assert result["value"].iloc[0] == 20


class TestBackendFactory:
    """Test the get_backend factory function."""

    def test_sqlite_default(self) -> None:
        """Default backend should be SQLite."""
        backend = get_backend("sqlite")
        assert isinstance(backend, SQLiteBackend)

    def test_unknown_backend_raises(self) -> None:
        """Unknown backend type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown DB backend"):
            get_backend("mysql")

    def test_postgres_not_implemented(self) -> None:
        """PostgreSQL should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            get_backend("postgres")
