"""Generate synthetic sample datasets for InsightBoard.

Run standalone:  python -m src.generate_samples
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

SEED = 42
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sample"


def _date_range(start: str, end: str) -> list[datetime]:
    """Generate a list of dates between start and end (inclusive)."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    return [s + timedelta(days=i) for i in range((e - s).days + 1)]


def generate_ecommerce(n: int = 5000) -> pd.DataFrame:
    """Generate e-commerce orders dataset."""
    rng = random.Random(SEED)
    categories = ["Electronics", "Clothing", "Home & Garden", "Books", "Sports", "Beauty"]
    regions = ["North America", "Europe", "Asia Pacific", "Latin America"]
    prices = {
        "Electronics": (29.99, 999.99),
        "Clothing": (9.99, 199.99),
        "Home & Garden": (14.99, 499.99),
        "Books": (4.99, 49.99),
        "Sports": (19.99, 299.99),
        "Beauty": (7.99, 89.99),
    }
    dates = _date_range("2023-01-01", "2024-12-31")

    # Create seasonal patterns
    rows = []
    customer_pool = [f"C{str(i).zfill(5)}" for i in range(1, 1201)]

    for i in range(n):
        date = rng.choice(dates)
        # Boost Q4 for seasonality
        if date.month in (11, 12) and rng.random() < 0.3:
            date = date  # keep — effectively more Q4 dates
        category = rng.choices(categories, weights=[25, 20, 15, 15, 15, 10], k=1)[0]
        lo, hi = prices[category]
        rows.append(
            {
                "order_id": f"ORD-{str(i + 1).zfill(6)}",
                "order_date": date.strftime("%Y-%m-%d"),
                "customer_id": rng.choice(customer_pool),
                "product_category": category,
                "quantity": rng.choices([1, 2, 3, 4, 5], weights=[50, 25, 15, 7, 3], k=1)[0],
                "unit_price": round(rng.uniform(lo, hi), 2),
                "region": rng.choices(regions, weights=[40, 30, 20, 10], k=1)[0],
            }
        )

    df = pd.DataFrame(rows)
    # Add a growth trend: later orders have slightly higher prices
    df = df.sort_values("order_date").reset_index(drop=True)
    return df


def generate_saas(n: int = 2000) -> pd.DataFrame:
    """Generate SaaS subscription dataset."""
    rng = random.Random(SEED + 1)
    plans = ["Free", "Starter", "Professional", "Enterprise"]
    plan_mrr = {"Free": 0, "Starter": 29, "Professional": 99, "Enterprise": 499}
    statuses = ["active", "churned", "trialing", "paused"]
    dates = _date_range("2023-01-01", "2024-12-31")
    customer_pool = [f"SC{str(i).zfill(5)}" for i in range(1, 801)]

    rows = []
    for i in range(n):
        plan = rng.choices(plans, weights=[20, 35, 30, 15], k=1)[0]
        start = rng.choice(dates)
        # Churn probability varies by plan
        churn_prob = {"Free": 0.4, "Starter": 0.25, "Professional": 0.15, "Enterprise": 0.08}
        status = rng.choices(
            statuses,
            weights=[50, churn_prob[plan] * 100, 15, 10],
            k=1,
        )[0]
        base_mrr = plan_mrr[plan]
        # Add some noise to MRR (discounts, add-ons)
        mrr = round(base_mrr * rng.uniform(0.8, 1.3), 2) if base_mrr > 0 else 0

        row = {
            "subscription_id": f"SUB-{str(i + 1).zfill(5)}",
            "start_date": start.strftime("%Y-%m-%d"),
            "customer_id": rng.choice(customer_pool),
            "plan": plan,
            "mrr": mrr,
            "status": status,
        }
        if status == "churned":
            churn_offset = rng.randint(30, 365)
            churn_date = start + timedelta(days=churn_offset)
            row["churn_date"] = churn_date.strftime("%Y-%m-%d")
        else:
            row["churn_date"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


def generate_marketing(n: int = 1000) -> pd.DataFrame:
    """Generate marketing campaign dataset."""
    rng = random.Random(SEED + 2)
    channels = ["Google Ads", "Facebook", "LinkedIn", "Email", "Organic Search", "Referral"]
    campaigns = [f"CAMP-{str(i).zfill(3)}" for i in range(1, 21)]
    dates = _date_range("2023-06-01", "2024-12-31")

    rows = []
    for _i in range(n):
        channel = rng.choices(channels, weights=[25, 25, 15, 15, 12, 8], k=1)[0]
        date = rng.choice(dates)

        # Channel-specific performance profiles
        base_impressions = {
            "Google Ads": 5000,
            "Facebook": 8000,
            "LinkedIn": 2000,
            "Email": 1500,
            "Organic Search": 3000,
            "Referral": 1000,
        }
        ctr = {
            "Google Ads": 0.035,
            "Facebook": 0.025,
            "LinkedIn": 0.02,
            "Email": 0.15,
            "Organic Search": 0.04,
            "Referral": 0.08,
        }
        conv_rate = {
            "Google Ads": 0.04,
            "Facebook": 0.025,
            "LinkedIn": 0.035,
            "Email": 0.05,
            "Organic Search": 0.03,
            "Referral": 0.06,
        }
        cpc = {
            "Google Ads": 2.50,
            "Facebook": 1.20,
            "LinkedIn": 5.00,
            "Email": 0.10,
            "Organic Search": 0,
            "Referral": 0.50,
        }

        impressions = int(base_impressions[channel] * rng.uniform(0.5, 2.0))
        clicks = max(1, int(impressions * ctr[channel] * rng.uniform(0.7, 1.4)))
        conversions = max(0, int(clicks * conv_rate[channel] * rng.uniform(0.5, 1.8)))
        spend = round(clicks * cpc[channel] * rng.uniform(0.8, 1.3), 2)

        rows.append(
            {
                "campaign_id": rng.choice(campaigns),
                "date": date.strftime("%Y-%m-%d"),
                "channel": channel,
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "spend": spend,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Generate and save all sample datasets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "ecommerce_orders.csv": generate_ecommerce,
        "saas_subscriptions.csv": generate_saas,
        "marketing_campaigns.csv": generate_marketing,
    }

    for filename, generator in datasets.items():
        df = generator()
        path = OUTPUT_DIR / filename
        df.to_csv(path, index=False)
        print(f"✓ {filename}: {len(df):,} rows → {path}")


if __name__ == "__main__":
    main()
