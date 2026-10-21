"""Tests for the validation module."""

from __future__ import annotations

import pandas as pd

from src.validation import (
    QualityReport,
    validate_generic,
    validate_with_schema,
)


class TestQualityReport:
    """Test the QualityReport data class."""

    def test_empty_report_no_errors(self) -> None:
        report = QualityReport(dataset_name="test", total_rows=100, total_columns=5)
        assert not report.has_errors
        assert not report.has_warnings
        assert report.error_count == 0
        assert "✅" in report.summary_text()

    def test_report_with_errors(self) -> None:
        from src.validation import QualityIssue

        report = QualityReport(
            dataset_name="test",
            total_rows=100,
            total_columns=5,
            issues=[
                QualityIssue(severity="error", column="id", check="null", message="nulls found"),
            ],
        )
        assert report.has_errors
        assert report.error_count == 1
        assert "🔴" in report.summary_text()


class TestValidateGeneric:
    """Test generic validation on arbitrary DataFrames."""

    def test_empty_df_flags_error(self) -> None:
        df = pd.DataFrame()
        report = validate_generic(df, "empty_test")
        assert report.has_errors
        assert any("empty" in i.message.lower() for i in report.issues)

    def test_nulls_detected(self) -> None:
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
        report = validate_generic(df)
        null_issues = [i for i in report.issues if i.check == "null_check"]
        assert len(null_issues) >= 1

    def test_duplicates_detected(self) -> None:
        df = pd.DataFrame(
            {
                "order_id": ["A", "A", "B"],
                "value": [1, 1, 2],
            }
        )
        report = validate_generic(df)
        dup_issues = [i for i in report.issues if "duplicate" in i.check]
        assert len(dup_issues) >= 1

    def test_date_parsing_errors(self) -> None:
        df = pd.DataFrame({"order_date": ["2024-01-01", "not-a-date", "2024-03-15"]})
        report = validate_generic(df)
        date_issues = [i for i in report.issues if i.check == "date_parsing"]
        assert len(date_issues) >= 1

    def test_clean_df_passes(self) -> None:
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Carol"],
                "score": [90, 85, 92],
            }
        )
        report = validate_generic(df)
        assert not report.has_errors


class TestValidateWithSchema:
    """Test schema-aware validation."""

    def test_valid_ecommerce(self, sample_ecommerce_df: pd.DataFrame) -> None:
        report = validate_with_schema(sample_ecommerce_df, "ecommerce_orders")
        assert not report.has_errors

    def test_invalid_ecommerce_missing_column(self) -> None:
        df = pd.DataFrame({"order_id": ["1"], "order_date": ["2024-01-01"]})
        report = validate_with_schema(df, "ecommerce_orders")
        # Should flag missing required columns
        assert report.has_errors or report.has_warnings

    def test_unknown_schema_falls_back(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        report = validate_with_schema(df, "unknown_dataset")
        # Should not crash — falls back to generic
        assert report.dataset_name == "unknown_dataset"
