"""Tests for the transformation layer."""

from __future__ import annotations

import pandas as pd

from src.transforms import (
    add_cohort,
    add_revenue,
    coerce_numerics,
    detect_categorical_columns,
    detect_date_column,
    detect_numeric_columns,
    parse_dates,
    transform_dataset,
    transform_ecommerce,
    transform_marketing,
    transform_saas,
)


class TestParseDates:
    """Test date parsing and period bucket creation."""

    def test_basic_parsing(self) -> None:
        df = pd.DataFrame({"order_date": ["2024-01-15", "2024-03-20", "2024-06-01"]})
        result = parse_dates(df, "order_date")

        assert "_parsed_date" in result.columns
        assert "period_day" in result.columns
        assert "period_week" in result.columns
        assert "period_month" in result.columns

    def test_period_month_format(self) -> None:
        df = pd.DataFrame({"order_date": ["2024-01-15"]})
        result = parse_dates(df, "order_date")
        assert result["period_month"].iloc[0] == "2024-01"

    def test_unparseable_dates_dropped(self) -> None:
        df = pd.DataFrame({"order_date": ["2024-01-01", "garbage", "2024-03-15"]})
        result = parse_dates(df, "order_date")
        assert len(result) == 2  # garbage row dropped

    def test_empty_df(self) -> None:
        df = pd.DataFrame({"order_date": pd.Series([], dtype=str)})
        result = parse_dates(df, "order_date")
        assert len(result) == 0


class TestAddRevenue:
    """Test revenue calculation."""

    def test_revenue_computed(self) -> None:
        df = pd.DataFrame({"quantity": [2, 3], "unit_price": [10.0, 20.0]})
        result = add_revenue(df)
        assert "revenue" in result.columns
        assert result["revenue"].tolist() == [20.0, 60.0]

    def test_missing_columns_no_crash(self) -> None:
        df = pd.DataFrame({"foo": [1, 2]})
        result = add_revenue(df)
        assert "revenue" not in result.columns  # gracefully skipped

    def test_string_quantities_coerced(self) -> None:
        df = pd.DataFrame({"quantity": ["2", "3"], "unit_price": ["10.0", "20.0"]})
        result = add_revenue(df)
        assert result["revenue"].tolist() == [20.0, 60.0]


class TestAddCohort:
    """Test cohort assignment."""

    def test_cohort_assigned(self) -> None:
        df = pd.DataFrame(
            {
                "customer_id": ["C1", "C1", "C2", "C2"],
                "_parsed_date": pd.to_datetime(
                    ["2024-01-15", "2024-03-20", "2024-02-01", "2024-04-10"]
                ),
            }
        )
        result = add_cohort(df)
        assert "cohort_month" in result.columns
        # C1's first purchase is Jan 2024
        c1_rows = result[result["customer_id"] == "C1"]
        assert (c1_rows["cohort_month"] == "2024-01").all()

    def test_missing_columns_no_crash(self) -> None:
        df = pd.DataFrame({"foo": [1]})
        result = add_cohort(df)
        assert "cohort_month" not in result.columns


class TestCoerceNumerics:
    """Test numeric coercion."""

    def test_string_to_numeric(self) -> None:
        df = pd.DataFrame({"amount": ["100", "200.5", "not_a_number"]})
        result = coerce_numerics(df, ["amount"])
        assert result["amount"].tolist() == [100.0, 200.5, 0.0]

    def test_auto_detect(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = coerce_numerics(df)
        assert pd.api.types.is_numeric_dtype(result["a"])


class TestTransformPipelines:
    """Test dataset-specific transform pipelines."""

    def test_ecommerce_pipeline(self, sample_ecommerce_df: pd.DataFrame) -> None:
        result = transform_ecommerce(sample_ecommerce_df)
        assert "revenue" in result.columns
        assert "period_month" in result.columns
        assert "cohort_month" in result.columns
        assert result["revenue"].iloc[0] == 299.99  # 1 × 299.99

    def test_saas_pipeline(self, sample_saas_df: pd.DataFrame) -> None:
        result = transform_saas(sample_saas_df)
        assert "period_month" in result.columns
        assert pd.api.types.is_numeric_dtype(result["mrr"])

    def test_marketing_pipeline(self, sample_marketing_df: pd.DataFrame) -> None:
        result = transform_marketing(sample_marketing_df)
        assert "ctr" in result.columns
        assert "conversion_rate" in result.columns
        assert "cost_per_conversion" in result.columns

    def test_generic_fallback(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-02-01"],
                "value": [100, 200],
            }
        )
        result = transform_dataset(df, "unknown_dataset")
        assert "_parsed_date" in result.columns  # date detected and parsed


class TestDetectors:
    """Test column detection heuristics."""

    def test_detect_date_column(self) -> None:
        df = pd.DataFrame({"order_date": ["2024-01-01"], "amount": [100]})
        assert detect_date_column(df) == "order_date"

    def test_detect_date_column_none(self) -> None:
        df = pd.DataFrame({"foo": [1], "bar": ["x"]})
        assert detect_date_column(df) is None

    def test_detect_categorical(self) -> None:
        df = pd.DataFrame(
            {
                "category": ["A", "B", "C"] * 10,
                "order_id": [f"id_{i}" for i in range(30)],
                "value": range(30),
            }
        )
        cats = detect_categorical_columns(df)
        assert "category" in cats
        assert "order_id" not in cats  # high cardinality relative to _id suffix
        assert "value" not in cats

    def test_detect_numeric(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": ["3.0", "4.0"]})
        nums = detect_numeric_columns(df)
        assert "a" in nums
        assert "c" in nums
        assert "b" not in nums
