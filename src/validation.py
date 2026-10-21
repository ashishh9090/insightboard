"""Data validation with pandera.

Defines schemas for each bundled sample dataset, plus a generic validator
for user-uploaded files. Returns a structured quality report rather than
raising exceptions — the UI decides whether to warn or block.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from src.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Quality report data class
# ---------------------------------------------------------------------------


@dataclass
class QualityIssue:
    """A single data-quality finding."""

    severity: str  # "error" | "warning" | "info"
    column: str | None
    check: str
    message: str
    row_count: int = 0


@dataclass
class QualityReport:
    """Aggregated quality report for a DataFrame."""

    dataset_name: str
    total_rows: int
    total_columns: int
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def summary_text(self) -> str:
        """One-line human summary."""
        parts = [f"{self.total_rows:,} rows × {self.total_columns} columns"]
        if self.error_count:
            parts.append(f"🔴 {self.error_count} error(s)")
        if self.warning_count:
            parts.append(f"🟡 {self.warning_count} warning(s)")
        if not self.issues:
            parts.append("✅ All checks passed")
        return " · ".join(parts)


# ---------------------------------------------------------------------------
# Sample dataset schemas
# ---------------------------------------------------------------------------

ECOMMERCE_SCHEMA = DataFrameSchema(
    columns={
        "order_id": Column(str, nullable=False, unique=True),
        "order_date": Column(str, nullable=False),
        "customer_id": Column(str, nullable=False),
        "product_category": Column(str, nullable=False),
        "quantity": Column(int, Check.greater_than(0), nullable=False, coerce=True),
        "unit_price": Column(float, Check.greater_than(0), nullable=False, coerce=True),
        "region": Column(str, nullable=False),
    },
    name="ecommerce_orders",
    strict=False,  # allow extra columns
    coerce=True,
)

SAAS_SCHEMA = DataFrameSchema(
    columns={
        "subscription_id": Column(str, nullable=False, unique=True),
        "start_date": Column(str, nullable=False),
        "customer_id": Column(str, nullable=False),
        "plan": Column(str, nullable=False),
        "mrr": Column(float, Check.greater_than_or_equal_to(0), nullable=False, coerce=True),
        "status": Column(
            str,
            Check.isin(["active", "churned", "trialing", "paused"]),
            nullable=False,
        ),
    },
    name="saas_subscriptions",
    strict=False,
    coerce=True,
)

MARKETING_SCHEMA = DataFrameSchema(
    columns={
        "campaign_id": Column(str, nullable=False),
        "date": Column(str, nullable=False),
        "channel": Column(str, nullable=False),
        "impressions": Column(int, Check.greater_than_or_equal_to(0), nullable=False, coerce=True),
        "clicks": Column(int, Check.greater_than_or_equal_to(0), nullable=False, coerce=True),
        "conversions": Column(int, Check.greater_than_or_equal_to(0), nullable=False, coerce=True),
        "spend": Column(float, Check.greater_than_or_equal_to(0), nullable=False, coerce=True),
    },
    name="marketing_campaigns",
    strict=False,
    coerce=True,
)

SCHEMA_REGISTRY: dict[str, DataFrameSchema] = {
    "ecommerce_orders": ECOMMERCE_SCHEMA,
    "saas_subscriptions": SAAS_SCHEMA,
    "marketing_campaigns": MARKETING_SCHEMA,
}


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def _check_nulls(df: pd.DataFrame, report: QualityReport) -> None:
    """Flag columns with null values."""
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            pct = null_count / len(df) * 100
            severity = "error" if pct > 50 else "warning"
            report.issues.append(
                QualityIssue(
                    severity=severity,
                    column=col,
                    check="null_check",
                    message=f"{null_count:,} null values ({pct:.1f}%)",
                    row_count=null_count,
                )
            )


def _check_duplicates(df: pd.DataFrame, id_column: str | None, report: QualityReport) -> None:
    """Flag duplicate rows or duplicate IDs."""
    full_dupes = int(df.duplicated().sum())
    if full_dupes > 0:
        report.issues.append(
            QualityIssue(
                severity="warning",
                column=None,
                check="duplicate_rows",
                message=f"{full_dupes:,} fully duplicated rows",
                row_count=full_dupes,
            )
        )
    if id_column and id_column in df.columns:
        id_dupes = int(df[id_column].duplicated().sum())
        if id_dupes > 0:
            report.issues.append(
                QualityIssue(
                    severity="error",
                    column=id_column,
                    check="duplicate_ids",
                    message=f"{id_dupes:,} duplicate IDs in '{id_column}'",
                    row_count=id_dupes,
                )
            )


def _check_date_parsing(df: pd.DataFrame, date_columns: list[str], report: QualityReport) -> None:
    """Verify that date columns can be parsed."""
    for col in date_columns:
        if col not in df.columns:
            continue
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            failed = int(parsed.isna().sum() - df[col].isna().sum())
            if failed > 0:
                report.issues.append(
                    QualityIssue(
                        severity="error",
                        column=col,
                        check="date_parsing",
                        message=f"{failed:,} values could not be parsed as dates",
                        row_count=failed,
                    )
                )
        except Exception:
            report.issues.append(
                QualityIssue(
                    severity="error",
                    column=col,
                    check="date_parsing",
                    message="Column could not be parsed as dates at all",
                )
            )


def _detect_date_columns(df: pd.DataFrame) -> list[str]:
    """Heuristic: find columns likely to contain dates."""
    candidates = []
    date_keywords = {"date", "time", "timestamp", "created", "updated", "start", "end"}
    for col in df.columns:
        if any(kw in col.lower() for kw in date_keywords):
            candidates.append(col)
    return candidates


def _detect_id_column(df: pd.DataFrame) -> str | None:
    """Heuristic: find the primary ID column."""
    for col in df.columns:
        if (col.lower().endswith("_id") or col.lower() == "id") and (
            df[col].nunique() > len(df) * 0.9
        ):
            return col
    return None


def validate_with_schema(df: pd.DataFrame, schema_name: str) -> QualityReport:
    """Validate a DataFrame against a known pandera schema.

    Args:
        df: The DataFrame to validate.
        schema_name: Key into ``SCHEMA_REGISTRY``.

    Returns:
        A ``QualityReport`` with all findings.
    """
    report = QualityReport(
        dataset_name=schema_name,
        total_rows=len(df),
        total_columns=len(df.columns),
    )

    schema = SCHEMA_REGISTRY.get(schema_name)
    if schema is None:
        report.issues.append(
            QualityIssue(
                severity="warning",
                column=None,
                check="schema_lookup",
                message=f"No registered schema for '{schema_name}'; running generic checks only.",
            )
        )
        return validate_generic(df, schema_name)

    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        for _, row in exc.failure_cases.iterrows():
            report.issues.append(
                QualityIssue(
                    severity="error",
                    column=str(row.get("column", "")),
                    check=str(row.get("check", "schema")),
                    message=str(row.get("failure_case", "Validation failed")),
                    row_count=1,
                )
            )

    # Additional checks beyond pandera
    date_cols = _detect_date_columns(df)
    _check_date_parsing(df, date_cols, report)

    logger.info(
        "schema validation complete",
        extra={
            "dataset": schema_name,
            "errors": report.error_count,
            "warnings": report.warning_count,
        },
    )
    return report


def validate_generic(df: pd.DataFrame, name: str = "uploaded_data") -> QualityReport:
    """Run generic quality checks on an arbitrary DataFrame.

    Used for user uploads that don't match a known schema.

    Args:
        df: The DataFrame to validate.
        name: Display name for the dataset.

    Returns:
        A ``QualityReport`` with all findings.
    """
    report = QualityReport(
        dataset_name=name,
        total_rows=len(df),
        total_columns=len(df.columns),
    )

    if df.empty:
        report.issues.append(
            QualityIssue(
                severity="error",
                column=None,
                check="empty_check",
                message="Dataset is empty (0 rows).",
            )
        )
        return report

    _check_nulls(df, report)
    id_col = _detect_id_column(df)
    _check_duplicates(df, id_col, report)
    date_cols = _detect_date_columns(df)
    _check_date_parsing(df, date_cols, report)

    # Type-mix check: flag columns with mixed types
    for col in df.columns:
        types = df[col].dropna().apply(type).nunique()
        if types > 1:
            report.issues.append(
                QualityIssue(
                    severity="warning",
                    column=col,
                    check="mixed_types",
                    message=f"Column has {types} different Python types — may need cleaning.",
                )
            )

    logger.info(
        "generic validation complete",
        extra={
            "dataset": name,
            "errors": report.error_count,
            "warnings": report.warning_count,
        },
    )
    return report
