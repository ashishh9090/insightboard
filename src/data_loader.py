"""Data loader — handles CSV/XLSX upload and bundled sample datasets.

Responsibilities:
  1. Read CSV or XLSX into a DataFrame
  2. Run validation (schema-aware for samples, generic for uploads)
  3. Insert into the database backend
  4. Return the QualityReport so the UI can display it
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.config import settings
from src.db import DatabaseBackend
from src.logging_setup import get_logger, metrics
from src.validation import QualityReport, validate_generic, validate_with_schema

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sample dataset registry
# ---------------------------------------------------------------------------

SAMPLE_DATASETS: dict[str, dict[str, str]] = {
    "ecommerce_orders": {
        "label": "🛒 E-Commerce Orders",
        "file": "ecommerce_orders.csv",
        "description": "~5 000 orders with categories, regions, and revenue.",
    },
    "saas_subscriptions": {
        "label": "💳 SaaS Subscriptions",
        "file": "saas_subscriptions.csv",
        "description": "~2 000 subscription events with MRR and churn.",
    },
    "marketing_campaigns": {
        "label": "📣 Marketing Campaigns",
        "file": "marketing_campaigns.csv",
        "description": "~1 000 daily campaign metrics across channels.",
    },
}


def list_sample_datasets() -> dict[str, dict[str, str]]:
    """Return metadata for all bundled sample datasets."""
    return SAMPLE_DATASETS


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_file(source: str | Path | BinaryIO, file_name: str) -> pd.DataFrame:
    """Read a CSV or XLSX into a DataFrame based on extension.

    Args:
        source: File path or file-like object.
        file_name: Original filename (used to detect format).

    Returns:
        Raw DataFrame (no cleaning applied yet).

    Raises:
        ValueError: If the file format is unsupported.
    """
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source, dtype=str)  # read as strings; validation coerces
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(source, dtype=str, engine="openpyxl")
    raise ValueError(f"Unsupported file format: {suffix}. Please upload CSV or XLSX.")


def load_sample_dataset(
    dataset_key: str,
    db: DatabaseBackend,
) -> tuple[pd.DataFrame, QualityReport]:
    """Load a bundled sample dataset, validate, and insert into the DB.

    Args:
        dataset_key: Key from ``SAMPLE_DATASETS``.
        db: Database backend to insert into.

    Returns:
        Tuple of (DataFrame, QualityReport).

    Raises:
        FileNotFoundError: If the sample file is missing.
        KeyError: If ``dataset_key`` is not recognized.
    """
    if dataset_key not in SAMPLE_DATASETS:
        raise KeyError(f"Unknown sample dataset: {dataset_key!r}")

    meta = SAMPLE_DATASETS[dataset_key]
    file_path = settings.SAMPLE_DATA_DIR / meta["file"]

    if not file_path.exists():
        raise FileNotFoundError(f"Sample data file not found: {file_path}")

    t0 = time.perf_counter()
    df = _read_file(file_path, meta["file"])
    load_ms = (time.perf_counter() - t0) * 1000
    metrics.record_time("data_load", load_ms)

    # Validate with the registered schema
    report = validate_with_schema(df, dataset_key)

    # Insert into DB
    db.insert_dataframe(df, dataset_key)
    metrics.set_timestamp("data_freshness")

    logger.info(
        "sample dataset loaded",
        extra={
            "dataset": dataset_key,
            "rows": len(df),
            "load_ms": round(load_ms, 1),
            "errors": report.error_count,
        },
    )
    return df, report


def load_uploaded_file(
    file_obj: BinaryIO,
    file_name: str,
    db: DatabaseBackend,
    table_name: str | None = None,
) -> tuple[pd.DataFrame, QualityReport]:
    """Load a user-uploaded file, validate generically, and insert into DB.

    Args:
        file_obj: Uploaded file-like object.
        file_name: Original filename.
        db: Database backend to insert into.
        table_name: Table name override. Defaults to sanitized filename.

    Returns:
        Tuple of (DataFrame, QualityReport).
    """
    t0 = time.perf_counter()

    # Sanitize table name
    if table_name is None:
        table_name = Path(file_name).stem.lower().replace(" ", "_").replace("-", "_")
        # Remove non-alphanumeric chars (except underscore)
        table_name = "".join(c for c in table_name if c.isalnum() or c == "_")
        if not table_name:
            table_name = "uploaded_data"

    df = _read_file(file_obj, file_name)
    load_ms = (time.perf_counter() - t0) * 1000
    metrics.record_time("data_load", load_ms)

    # Check if it matches a known schema
    if table_name in SAMPLE_DATASETS:
        report = validate_with_schema(df, table_name)
    else:
        report = validate_generic(df, table_name)

    # Insert even if there are warnings (but not if empty)
    if not report.has_errors or report.total_rows > 0:
        db.insert_dataframe(df, table_name)
        metrics.set_timestamp("data_freshness")

    logger.info(
        "uploaded file loaded",
        extra={
            "file": file_name,
            "table": table_name,
            "rows": len(df),
            "load_ms": round(load_ms, 1),
            "errors": report.error_count,
        },
    )
    return df, report
