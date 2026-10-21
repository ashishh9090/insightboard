"""Tests for the analytical summary generator."""

from __future__ import annotations

import pandas as pd

from src.kpis import compute_kpis
from src.summary import (
    _biggest_mover,
    _detect_anomalies,
    _trend_direction,
    generate_rule_based_summary,
)
from src.transforms import transform_ecommerce


class TestTrendDirection:
    """Test trend detection via linear regression."""

    def test_upward_trend(self) -> None:
        df = pd.DataFrame(
            {
                "period_month": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"],
                "revenue": [100, 120, 150, 180, 220],
            }
        )
        assert _trend_direction(df, "period_month", "revenue") == "upward"

    def test_downward_trend(self) -> None:
        df = pd.DataFrame(
            {
                "period_month": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"],
                "revenue": [200, 180, 150, 120, 80],
            }
        )
        assert _trend_direction(df, "period_month", "revenue") == "downward"

    def test_flat_trend(self) -> None:
        df = pd.DataFrame(
            {
                "period_month": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"],
                "revenue": [100, 100, 100, 100, 100],
            }
        )
        assert _trend_direction(df, "period_month", "revenue") == "flat"

    def test_insufficient_data(self) -> None:
        df = pd.DataFrame({"period_month": ["2024-01"], "revenue": [100]})
        assert _trend_direction(df, "period_month", "revenue") == "insufficient data"

    def test_missing_column(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert _trend_direction(df, "period_month", "revenue") == "flat"


class TestBiggestMover:
    """Test biggest mover detection."""

    def test_identifies_biggest_mover(self) -> None:
        df = pd.DataFrame(
            {
                "period_month": ["2024-01", "2024-01", "2024-02", "2024-02"],
                "revenue": [100, 50, 200, 40],
                "category": ["A", "B", "A", "B"],
            }
        )
        result = _biggest_mover(df, "period_month", "revenue", "category")
        assert result is not None
        name, change = result
        assert name == "A"  # A went from 100 → 200 (+100), B went from 50 → 40 (-10)
        assert change == 100

    def test_no_periods(self) -> None:
        df = pd.DataFrame({"period_month": ["2024-01"], "revenue": [100], "category": ["A"]})
        assert _biggest_mover(df, "period_month", "revenue", "category") is None


class TestDetectAnomalies:
    """Test anomaly detection."""

    def test_spike_detected(self) -> None:
        # Normal values with one huge spike
        values = [100, 102, 98, 101, 500, 99, 103]
        df = pd.DataFrame(
            {
                "period_month": [f"2024-0{i + 1}" for i in range(len(values))],
                "revenue": values,
            }
        )
        anomalies = _detect_anomalies(df, "period_month", "revenue", sigma=2.0)
        assert len(anomalies) >= 1
        assert any(a["direction"] == "spike" for a in anomalies)

    def test_no_anomalies(self) -> None:
        df = pd.DataFrame(
            {
                "period_month": [f"2024-0{i + 1}" for i in range(6)],
                "revenue": [100, 101, 99, 102, 98, 101],
            }
        )
        anomalies = _detect_anomalies(df, "period_month", "revenue", sigma=2.0)
        assert len(anomalies) == 0

    def test_too_few_periods(self) -> None:
        df = pd.DataFrame({"period_month": ["2024-01", "2024-02"], "revenue": [100, 200]})
        anomalies = _detect_anomalies(df, "period_month", "revenue")
        assert len(anomalies) == 0  # needs at least 5 periods


class TestRuleBasedSummary:
    """Test the full rule-based summary generation."""

    def test_produces_markdown(self, sample_ecommerce_df: pd.DataFrame) -> None:
        transformed = transform_ecommerce(sample_ecommerce_df)
        kpis = compute_kpis(transformed, "ecommerce_orders")
        summary = generate_rule_based_summary(transformed, kpis, "ecommerce_orders")
        assert isinstance(summary, str)
        assert len(summary) > 50
        assert "**Key Trend:**" in summary
        assert "**Key Metrics:**" in summary

    def test_handles_empty_df(self) -> None:
        df = pd.DataFrame(columns=["period_month", "revenue", "product_category"])
        kpis = []
        summary = generate_rule_based_summary(df, kpis, "ecommerce_orders")
        assert isinstance(summary, str)
        assert "Key Trend" in summary
