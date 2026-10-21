"""Tests for KPI calculations."""

from __future__ import annotations

import pandas as pd

from src.kpis import (
    KPIResult,
    _format_number,
    _period_delta,
    _split_periods,
    compute_kpis,
)
from src.transforms import transform_ecommerce, transform_marketing, transform_saas


class TestFormatNumber:
    """Test number formatting helper."""

    def test_millions(self) -> None:
        assert _format_number(1_500_000, prefix="$") == "$1.5M"

    def test_thousands(self) -> None:
        assert _format_number(42_000) == "42.0K"

    def test_small_number(self) -> None:
        assert _format_number(123, decimals=1) == "123.0"

    def test_with_suffix(self) -> None:
        assert _format_number(85, suffix="%") == "85%"


class TestPeriodDelta:
    """Test period-over-period delta calculation."""

    def test_positive_growth(self) -> None:
        delta, label = _period_delta(120, 100)
        assert delta == 20.0
        assert "↑" in label

    def test_negative_growth(self) -> None:
        delta, label = _period_delta(80, 100)
        assert delta == -20.0
        assert "↓" in label

    def test_zero_prior(self) -> None:
        delta, label = _period_delta(100, 0)
        assert delta is None
        assert "N/A" in label

    def test_no_change(self) -> None:
        delta, label = _period_delta(100, 100)
        assert delta == 0.0
        assert "→" in label


class TestSplitPeriods:
    """Test period splitting."""

    def test_even_split(self) -> None:
        df = pd.DataFrame(
            {
                "_parsed_date": pd.to_datetime(
                    ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
                ),
                "value": [10, 20, 30, 40],
            }
        )
        current, prior = _split_periods(df)
        assert len(prior) > 0
        assert len(current) > 0

    def test_empty_df(self) -> None:
        df = pd.DataFrame({"_parsed_date": pd.Series([], dtype="datetime64[ns]")})
        current, prior = _split_periods(df)
        assert len(prior) == 0


class TestEcommerceKPIs:
    """Test e-commerce KPI computation."""

    def test_kpi_count(self, sample_ecommerce_df: pd.DataFrame) -> None:
        transformed = transform_ecommerce(sample_ecommerce_df)
        kpis = compute_kpis(transformed, "ecommerce_orders")
        assert len(kpis) == 6

    def test_kpi_types(self, sample_ecommerce_df: pd.DataFrame) -> None:
        transformed = transform_ecommerce(sample_ecommerce_df)
        kpis = compute_kpis(transformed, "ecommerce_orders")
        for kpi in kpis:
            assert isinstance(kpi, KPIResult)
            assert kpi.label
            assert kpi.formatted

    def test_revenue_positive(self, sample_ecommerce_df: pd.DataFrame) -> None:
        transformed = transform_ecommerce(sample_ecommerce_df)
        kpis = compute_kpis(transformed, "ecommerce_orders")
        revenue_kpi = next(k for k in kpis if k.label == "Total Revenue")
        assert revenue_kpi.value > 0


class TestSaaSKPIs:
    """Test SaaS KPI computation."""

    def test_kpi_count(self, sample_saas_df: pd.DataFrame) -> None:
        transformed = transform_saas(sample_saas_df)
        kpis = compute_kpis(transformed, "saas_subscriptions")
        assert len(kpis) == 6

    def test_churn_rate_bounded(self, sample_saas_df: pd.DataFrame) -> None:
        transformed = transform_saas(sample_saas_df)
        kpis = compute_kpis(transformed, "saas_subscriptions")
        churn_kpi = next(k for k in kpis if k.label == "Churn Rate")
        assert 0 <= churn_kpi.value <= 100


class TestMarketingKPIs:
    """Test marketing KPI computation."""

    def test_kpi_count(self, sample_marketing_df: pd.DataFrame) -> None:
        transformed = transform_marketing(sample_marketing_df)
        kpis = compute_kpis(transformed, "marketing_campaigns")
        assert len(kpis) == 6

    def test_ctr_positive(self, sample_marketing_df: pd.DataFrame) -> None:
        transformed = transform_marketing(sample_marketing_df)
        kpis = compute_kpis(transformed, "marketing_campaigns")
        ctr_kpi = next(k for k in kpis if k.label == "Avg CTR")
        assert ctr_kpi.value > 0


class TestGenericKPIs:
    """Test generic KPI fallback."""

    def test_unknown_dataset(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        kpis = compute_kpis(df, "random_dataset")
        assert len(kpis) >= 2  # at least rows + columns
        row_kpi = next(k for k in kpis if k.label == "Total Rows")
        assert row_kpi.value == 3
