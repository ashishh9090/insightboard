"""Tests for the data loader."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from src.data_loader import (
    _read_file,
    list_sample_datasets,
    load_sample_dataset,
    load_uploaded_file,
)
from src.db import SQLiteBackend


class TestListSampleDatasets:
    """Test sample dataset listing."""

    def test_returns_all_datasets(self) -> None:
        samples = list_sample_datasets()
        assert "ecommerce_orders" in samples
        assert "saas_subscriptions" in samples
        assert "marketing_campaigns" in samples

    def test_metadata_fields(self) -> None:
        samples = list_sample_datasets()
        for _key, meta in samples.items():
            assert "label" in meta
            assert "file" in meta
            assert "description" in meta


class TestReadFile:
    """Test file reading."""

    def test_read_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b\n1,x\n2,y\n")
        df = _read_file(str(csv_path), "test.csv")
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_unsupported_format(self) -> None:
        with pytest.raises(ValueError, match="Unsupported file format"):
            _read_file(io.BytesIO(b"data"), "test.json")


class TestLoadUploadedFile:
    """Test uploaded file loading."""

    def test_csv_upload(self, tmp_db: SQLiteBackend) -> None:
        csv_data = "id,name,value\n1,alice,100\n2,bob,200\n"
        file_obj = io.BytesIO(csv_data.encode())
        df, report = load_uploaded_file(file_obj, "my_data.csv", tmp_db)
        assert len(df) == 2
        assert not report.has_errors
        assert "my_data" in tmp_db.list_tables()

    def test_sanitized_table_name(self, tmp_db: SQLiteBackend) -> None:
        csv_data = "x\n1\n"
        file_obj = io.BytesIO(csv_data.encode())
        df, report = load_uploaded_file(file_obj, "My Data File (2024).csv", tmp_db)
        # Table name should be sanitized
        tables = tmp_db.list_tables()
        assert any(t.isalnum() or "_" in t for t in tables)


class TestLoadSampleDataset:
    """Test sample dataset loading (requires generated data files)."""

    def test_unknown_key_raises(self, tmp_db: SQLiteBackend) -> None:
        with pytest.raises(KeyError, match="Unknown sample dataset"):
            load_sample_dataset("nonexistent", tmp_db)
