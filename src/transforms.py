"""Data transformation layer.

Cleans raw data and builds derived fields:
  - Date parsing and period buckets (day, week, month)
  - Revenue calculation (quantity × unit_price)
  - Cohort assignment (first-purchase month per customer)
  - Growth rates
  - Generic numeric/categorical detection for unknown schemas

This module is pure-function, stateless, and fully testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------------


def detect_date_column(df: pd.DataFrame) -> str | None:
    """Return the best date column name, or None."""
    date_keywords = ["date", "time", "timestamp", "created", "updated"]
    for col in df.columns:
        if any(kw in col.lower() for kw in date_keywords):
            try:
                pd.to_datetime(df[col].dropna().head(20), errors="raise")
                return col
            except Exception:
                continue
    return None


def detect_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that are (or can be coerced to) numeric."""
    result = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            result.append(col)
        else:
            try:
                pd.to_numeric(df[col].dropna().head(50), errors="raise")
                result.append(col)
            except (ValueError, TypeError):
                continue
    return result


def detect_categorical_columns(df: pd.DataFrame, max_cardinality: int = 50) -> list[str]:
    """Return string columns with low cardinality (good for filters/breakdowns)."""
    result = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if col.lower().endswith("_id") or col.lower() == "id":
            continue
        nunique = df[col].nunique()
        if 2 <= nunique <= max_cardinality:
            result.append(col)
    return result


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------


def parse_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Parse a date column and add period bucket columns.

    Adds: ``_parsed_date``, ``period_day``, ``period_week``, ``period_month``.

    Args:
        df: Input DataFrame.
        date_col: Name of the raw date column.

    Returns:
        DataFrame with new columns appended.
    """
    df = df.copy()
    df["_parsed_date"] = pd.to_datetime(df[date_col], errors="coerce")

    # Drop rows where date parsing completely failed
    failed = df["_parsed_date"].isna().sum()
    if failed > 0:
        logger.warning(f"Dropped {failed} rows with unparseable dates in '{date_col}'")
        df = df.dropna(subset=["_parsed_date"])

    df["period_day"] = df["_parsed_date"].dt.strftime("%Y-%m-%d")
    df["period_week"] = df["_parsed_date"].dt.strftime("%Y-W%U")
    df["period_month"] = df["_parsed_date"].dt.strftime("%Y-%m")
    return df


def add_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Compute revenue = quantity × unit_price if both columns exist."""
    df = df.copy()
    if "quantity" in df.columns and "unit_price" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0)
        df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)
    return df


def add_cohort(df: pd.DataFrame, customer_col: str = "customer_id") -> pd.DataFrame:
    """Assign each customer to their first-purchase month cohort.

    Args:
        df: DataFrame with ``_parsed_date`` and a customer column.
        customer_col: Column identifying unique customers.

    Returns:
        DataFrame with ``cohort_month`` column.
    """
    df = df.copy()
    if customer_col not in df.columns or "_parsed_date" not in df.columns:
        return df

    first_purchase = (
        df.groupby(customer_col)["_parsed_date"].min().dt.strftime("%Y-%m").rename("cohort_month")
    )
    df = df.merge(first_purchase, on=customer_col, how="left")
    return df


def coerce_numerics(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Coerce specified columns to numeric, filling failures with 0.

    Args:
        df: Input DataFrame.
        columns: Columns to coerce. If None, auto-detects.

    Returns:
        DataFrame with numeric columns coerced.
    """
    df = df.copy()
    cols = columns or detect_numeric_columns(df)
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


# ---------------------------------------------------------------------------
# Dataset-specific transform pipelines
# ---------------------------------------------------------------------------


def transform_ecommerce(df: pd.DataFrame) -> pd.DataFrame:
    """Full transform pipeline for the e-commerce orders dataset."""
    df = parse_dates(df, "order_date")
    df = coerce_numerics(df, ["quantity", "unit_price"])
    df = add_revenue(df)
    df = add_cohort(df, "customer_id")
    return df


def transform_saas(df: pd.DataFrame) -> pd.DataFrame:
    """Full transform pipeline for the SaaS subscriptions dataset."""
    df = parse_dates(df, "start_date")
    df = coerce_numerics(df, ["mrr"])
    if "churn_date" in df.columns:
        df["churn_date_parsed"] = pd.to_datetime(df["churn_date"], errors="coerce")
        df["days_to_churn"] = (df["churn_date_parsed"] - df["_parsed_date"]).dt.days
    return df


def transform_marketing(df: pd.DataFrame) -> pd.DataFrame:
    """Full transform pipeline for the marketing campaigns dataset."""
    df = parse_dates(df, "date")
    df = coerce_numerics(df, ["impressions", "clicks", "conversions", "spend"])
    df["ctr"] = (df["clicks"] / df["impressions"].replace(0, np.nan) * 100).round(2)
    df["conversion_rate"] = (df["conversions"] / df["clicks"].replace(0, np.nan) * 100).round(2)
    df["cost_per_conversion"] = (df["spend"] / df["conversions"].replace(0, np.nan)).round(2)
    return df


# Transform registry keyed by dataset name
TRANSFORM_REGISTRY: dict[str, callable] = {
    "ecommerce_orders": transform_ecommerce,
    "saas_subscriptions": transform_saas,
    "marketing_campaigns": transform_marketing,
}


def transform_dataset(df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """Run the appropriate transform pipeline for a dataset.

    Falls back to a generic pipeline (parse dates + coerce numerics) for
    unknown datasets.

    Args:
        df: Raw DataFrame.
        dataset_key: Dataset identifier.

    Returns:
        Transformed DataFrame with derived fields.
    """
    if dataset_key in TRANSFORM_REGISTRY:
        result = TRANSFORM_REGISTRY[dataset_key](df)
        logger.info(
            "dataset transformed",
            extra={"dataset": dataset_key, "rows": len(result), "cols": len(result.columns)},
        )
        return result

    # Generic fallback
    logger.info(f"No registered transform for '{dataset_key}'; running generic pipeline")
    date_col = detect_date_column(df)
    if date_col:
        df = parse_dates(df, date_col)
    df = coerce_numerics(df)
    if "quantity" in df.columns and "unit_price" in df.columns:
        df = add_revenue(df)
    return df
