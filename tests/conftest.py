"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.db import SQLiteBackend


@pytest.fixture
def sample_ecommerce_df() -> pd.DataFrame:
    """Minimal e-commerce orders DataFrame for testing."""
    return pd.DataFrame(
        {
            "order_id": [f"ORD-{i:04d}" for i in range(1, 21)],
            "order_date": [
                "2024-01-05",
                "2024-01-12",
                "2024-01-20",
                "2024-02-03",
                "2024-02-14",
                "2024-03-01",
                "2024-03-15",
                "2024-03-28",
                "2024-04-10",
                "2024-04-22",
                "2024-05-05",
                "2024-05-18",
                "2024-06-01",
                "2024-06-15",
                "2024-07-01",
                "2024-07-20",
                "2024-08-05",
                "2024-08-22",
                "2024-09-10",
                "2024-09-25",
            ],
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C001",
                "C004",
                "C002",
                "C005",
                "C003",
                "C001",
                "C006",
                "C007",
                "C002",
                "C008",
                "C005",
                "C009",
                "C001",
                "C010",
                "C003",
                "C002",
                "C011",
            ],
            "product_category": [
                "Electronics",
                "Clothing",
                "Books",
                "Electronics",
                "Home & Garden",
                "Clothing",
                "Sports",
                "Electronics",
                "Books",
                "Beauty",
                "Electronics",
                "Clothing",
                "Home & Garden",
                "Sports",
                "Beauty",
                "Electronics",
                "Books",
                "Clothing",
                "Electronics",
                "Sports",
            ],
            "quantity": [1, 2, 1, 3, 1, 2, 1, 1, 4, 2, 1, 3, 2, 1, 1, 2, 1, 2, 1, 3],
            "unit_price": [
                299.99,
                49.99,
                14.99,
                599.99,
                89.99,
                79.99,
                149.99,
                199.99,
                9.99,
                24.99,
                449.99,
                39.99,
                129.99,
                199.99,
                34.99,
                349.99,
                19.99,
                59.99,
                799.99,
                79.99,
            ],
            "region": [
                "North America",
                "Europe",
                "Asia Pacific",
                "North America",
                "Europe",
                "Asia Pacific",
                "North America",
                "Europe",
                "North America",
                "Latin America",
                "North America",
                "Europe",
                "Asia Pacific",
                "North America",
                "Europe",
                "North America",
                "Asia Pacific",
                "Europe",
                "North America",
                "Latin America",
            ],
        }
    )


@pytest.fixture
def sample_saas_df() -> pd.DataFrame:
    """Minimal SaaS subscriptions DataFrame for testing."""
    return pd.DataFrame(
        {
            "subscription_id": [f"SUB-{i:04d}" for i in range(1, 11)],
            "start_date": [
                "2024-01-10",
                "2024-01-15",
                "2024-02-01",
                "2024-02-20",
                "2024-03-05",
                "2024-04-01",
                "2024-05-10",
                "2024-06-01",
                "2024-07-15",
                "2024-08-01",
            ],
            "customer_id": [f"SC{i:04d}" for i in range(1, 11)],
            "plan": [
                "Starter",
                "Professional",
                "Enterprise",
                "Free",
                "Starter",
                "Professional",
                "Enterprise",
                "Starter",
                "Professional",
                "Free",
            ],
            "mrr": [29.0, 99.0, 499.0, 0.0, 35.0, 110.0, 520.0, 25.0, 95.0, 0.0],
            "status": [
                "active",
                "active",
                "active",
                "churned",
                "active",
                "churned",
                "active",
                "trialing",
                "active",
                "paused",
            ],
            "churn_date": [
                "",
                "",
                "",
                "2024-06-15",
                "",
                "2024-08-01",
                "",
                "",
                "",
                "",
            ],
        }
    )


@pytest.fixture
def sample_marketing_df() -> pd.DataFrame:
    """Minimal marketing campaigns DataFrame for testing."""
    return pd.DataFrame(
        {
            "campaign_id": ["CAMP-001"] * 5 + ["CAMP-002"] * 5,
            "date": [
                "2024-01-15",
                "2024-02-15",
                "2024-03-15",
                "2024-04-15",
                "2024-05-15",
                "2024-01-20",
                "2024-02-20",
                "2024-03-20",
                "2024-04-20",
                "2024-05-20",
            ],
            "channel": [
                "Google Ads",
                "Google Ads",
                "Google Ads",
                "Google Ads",
                "Google Ads",
                "Facebook",
                "Facebook",
                "Facebook",
                "Facebook",
                "Facebook",
            ],
            "impressions": [5000, 6000, 5500, 7000, 8000, 8000, 9000, 7500, 10000, 11000],
            "clicks": [175, 210, 192, 245, 280, 200, 225, 187, 250, 275],
            "conversions": [7, 8, 8, 10, 11, 5, 6, 5, 6, 7],
            "spend": [437.5, 525.0, 480.0, 612.5, 700.0, 240.0, 270.0, 224.4, 300.0, 330.0],
        }
    )


@pytest.fixture
def tmp_db(tmp_path: Path) -> SQLiteBackend:
    """Create a temporary SQLite database for testing."""
    db_path = str(tmp_path / "test.db")
    return SQLiteBackend(db_path=db_path)
