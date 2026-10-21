"""KPI calculations with period-over-period deltas.

Each KPI function takes a transformed DataFrame and returns a ``KPIResult``.
The module is dataset-aware: different datasets surface different KPIs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class KPIResult:
    """A single KPI value with its delta."""

    label: str
    value: float | int
    formatted: str
    delta: float | None = None  # percent change vs prior period
    delta_label: str = ""
    icon: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_number(
    value: float | int,
    prefix: str = "",
    suffix: str = "",
    decimals: int | None = None,
) -> str:
    """Human-friendly number formatting."""
    dec = (1 if abs(value) >= 1_000 else 0) if decimals is None else decimals
    if abs(value) >= 1_000_000:
        return f"{prefix}{value / 1_000_000:,.{dec}f}M{suffix}"
    if abs(value) >= 1_000:
        return f"{prefix}{value / 1_000:,.{dec}f}K{suffix}"
    return f"{prefix}{value:,.{dec}f}{suffix}"


def _period_delta(current: float, prior: float) -> tuple[float | None, str]:
    """Calculate percent change and a human label."""
    if prior == 0:
        return None, "N/A (no prior data)"
    delta = ((current - prior) / abs(prior)) * 100
    direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
    return round(delta, 1), f"{direction} {abs(delta):.1f}% vs prior period"


def _split_periods(
    df: pd.DataFrame,
    date_col: str = "_parsed_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a date-sorted DataFrame into current and prior halves.

    Uses the midpoint of the date range as the boundary — the second half
    is 'current', the first half is 'prior'.
    """
    if date_col not in df.columns or df.empty:
        return df, pd.DataFrame()

    dates = df[date_col].dropna()
    if dates.empty:
        return df, pd.DataFrame()

    mid = dates.min() + (dates.max() - dates.min()) / 2
    current = df[df[date_col] >= mid]
    prior = df[df[date_col] < mid]
    return current, prior


# ---------------------------------------------------------------------------
# E-commerce KPIs
# ---------------------------------------------------------------------------


def _ecommerce_kpis(df: pd.DataFrame) -> list[KPIResult]:
    """Compute KPIs for e-commerce orders."""
    current, prior = _split_periods(df)
    results = []

    # 1. Total Revenue
    rev_curr = current["revenue"].sum() if "revenue" in current.columns else 0
    rev_prior = prior["revenue"].sum() if "revenue" in prior.columns else 0
    delta, delta_label = _period_delta(rev_curr, rev_prior)
    results.append(
        KPIResult(
            label="Total Revenue",
            value=rev_curr,
            formatted=_format_number(rev_curr, prefix="$", decimals=1),
            delta=delta,
            delta_label=delta_label,
            icon="💰",
        )
    )

    # 2. Order Count
    orders_curr = len(current)
    orders_prior = len(prior)
    delta, delta_label = _period_delta(orders_curr, orders_prior)
    results.append(
        KPIResult(
            label="Orders",
            value=orders_curr,
            formatted=_format_number(orders_curr),
            delta=delta,
            delta_label=delta_label,
            icon="📦",
        )
    )

    # 3. Average Order Value
    aov_curr = rev_curr / orders_curr if orders_curr > 0 else 0
    aov_prior = rev_prior / orders_prior if orders_prior > 0 else 0
    delta, delta_label = _period_delta(aov_curr, aov_prior)
    results.append(
        KPIResult(
            label="Avg Order Value",
            value=aov_curr,
            formatted=_format_number(aov_curr, prefix="$", decimals=2),
            delta=delta,
            delta_label=delta_label,
            icon="🛒",
        )
    )

    # 4. Unique Customers
    cust_curr = current["customer_id"].nunique() if "customer_id" in current.columns else 0
    cust_prior = prior["customer_id"].nunique() if "customer_id" in prior.columns else 0
    delta, delta_label = _period_delta(cust_curr, cust_prior)
    results.append(
        KPIResult(
            label="Unique Customers",
            value=cust_curr,
            formatted=_format_number(cust_curr),
            delta=delta,
            delta_label=delta_label,
            icon="👥",
        )
    )

    # 5. Revenue Growth %
    growth = ((rev_curr - rev_prior) / abs(rev_prior) * 100) if rev_prior != 0 else 0
    results.append(
        KPIResult(
            label="Revenue Growth",
            value=growth,
            formatted=f"{growth:+.1f}%",
            icon="📈",
        )
    )

    # 6. Top Category Share
    if "product_category" in current.columns and "revenue" in current.columns and not current.empty:
        cat_rev = current.groupby("product_category")["revenue"].sum()
        top_cat = cat_rev.idxmax()
        top_share = (cat_rev.max() / cat_rev.sum() * 100) if cat_rev.sum() > 0 else 0
        results.append(
            KPIResult(
                label=f"Top Category ({top_cat})",
                value=top_share,
                formatted=f"{top_share:.1f}%",
                icon="🏆",
            )
        )
    else:
        results.append(
            KPIResult(
                label="Top Category",
                value=0,
                formatted="N/A",
                icon="🏆",
            )
        )

    return results


# ---------------------------------------------------------------------------
# SaaS KPIs
# ---------------------------------------------------------------------------


def _saas_kpis(df: pd.DataFrame) -> list[KPIResult]:
    """Compute KPIs for SaaS subscriptions."""
    current, prior = _split_periods(df)
    results = []

    # 1. Total MRR
    mrr_curr = current["mrr"].sum() if "mrr" in current.columns else 0
    mrr_prior = prior["mrr"].sum() if "mrr" in prior.columns else 0
    delta, delta_label = _period_delta(mrr_curr, mrr_prior)
    results.append(
        KPIResult(
            label="Total MRR",
            value=mrr_curr,
            formatted=_format_number(mrr_curr, prefix="$", decimals=1),
            delta=delta,
            delta_label=delta_label,
            icon="💰",
        )
    )

    # 2. Active Subscriptions
    active_curr = len(current[current["status"] == "active"]) if "status" in current.columns else 0
    active_prior = len(prior[prior["status"] == "active"]) if "status" in prior.columns else 0
    delta, delta_label = _period_delta(active_curr, active_prior)
    results.append(
        KPIResult(
            label="Active Subscriptions",
            value=active_curr,
            formatted=_format_number(active_curr),
            delta=delta,
            delta_label=delta_label,
            icon="✅",
        )
    )

    # 3. Churn Count
    churn_curr = len(current[current["status"] == "churned"]) if "status" in current.columns else 0
    churn_prior = len(prior[prior["status"] == "churned"]) if "status" in prior.columns else 0
    delta, delta_label = _period_delta(churn_curr, churn_prior)
    results.append(
        KPIResult(
            label="Churned",
            value=churn_curr,
            formatted=_format_number(churn_curr),
            delta=delta,
            delta_label=delta_label,
            icon="🔴",
        )
    )

    # 4. Churn Rate
    total_curr = len(current)
    churn_rate = (churn_curr / total_curr * 100) if total_curr > 0 else 0
    results.append(
        KPIResult(
            label="Churn Rate",
            value=churn_rate,
            formatted=f"{churn_rate:.1f}%",
            icon="📉",
        )
    )

    # 5. Avg MRR per Customer
    unique_curr = current["customer_id"].nunique() if "customer_id" in current.columns else 0
    avg_mrr = mrr_curr / unique_curr if unique_curr > 0 else 0
    results.append(
        KPIResult(
            label="Avg MRR / Customer",
            value=avg_mrr,
            formatted=_format_number(avg_mrr, prefix="$", decimals=2),
            icon="👤",
        )
    )

    # 6. Enterprise Share
    if "plan" in current.columns and "mrr" in current.columns and not current.empty:
        plan_mrr = current.groupby("plan")["mrr"].sum()
        ent_share = (
            (plan_mrr.get("Enterprise", 0) / plan_mrr.sum() * 100) if plan_mrr.sum() > 0 else 0
        )
        results.append(
            KPIResult(
                label="Enterprise MRR Share",
                value=ent_share,
                formatted=f"{ent_share:.1f}%",
                icon="🏢",
            )
        )
    else:
        results.append(
            KPIResult(
                label="Enterprise MRR Share",
                value=0,
                formatted="N/A",
                icon="🏢",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Marketing KPIs
# ---------------------------------------------------------------------------


def _marketing_kpis(df: pd.DataFrame) -> list[KPIResult]:
    """Compute KPIs for marketing campaigns."""
    current, prior = _split_periods(df)
    results = []

    # 1. Total Spend
    spend_curr = current["spend"].sum() if "spend" in current.columns else 0
    spend_prior = prior["spend"].sum() if "spend" in prior.columns else 0
    delta, delta_label = _period_delta(spend_curr, spend_prior)
    results.append(
        KPIResult(
            label="Total Spend",
            value=spend_curr,
            formatted=_format_number(spend_curr, prefix="$", decimals=1),
            delta=delta,
            delta_label=delta_label,
            icon="💸",
        )
    )

    # 2. Total Conversions
    conv_curr = current["conversions"].sum() if "conversions" in current.columns else 0
    conv_prior = prior["conversions"].sum() if "conversions" in prior.columns else 0
    delta, delta_label = _period_delta(conv_curr, conv_prior)
    results.append(
        KPIResult(
            label="Conversions",
            value=int(conv_curr),
            formatted=_format_number(conv_curr),
            delta=delta,
            delta_label=delta_label,
            icon="🎯",
        )
    )

    # 3. Cost per Conversion
    cpc_curr = spend_curr / conv_curr if conv_curr > 0 else 0
    cpc_prior = spend_prior / conv_prior if conv_prior > 0 else 0
    delta, delta_label = _period_delta(cpc_curr, cpc_prior)
    results.append(
        KPIResult(
            label="Cost / Conversion",
            value=cpc_curr,
            formatted=_format_number(cpc_curr, prefix="$", decimals=2),
            delta=delta,
            delta_label=delta_label,
            icon="💵",
        )
    )

    # 4. Total Clicks
    clicks_curr = current["clicks"].sum() if "clicks" in current.columns else 0
    results.append(
        KPIResult(
            label="Total Clicks",
            value=int(clicks_curr),
            formatted=_format_number(clicks_curr),
            icon="👆",
        )
    )

    # 5. Avg CTR
    impr_curr = current["impressions"].sum() if "impressions" in current.columns else 0
    avg_ctr = (clicks_curr / impr_curr * 100) if impr_curr > 0 else 0
    results.append(
        KPIResult(
            label="Avg CTR",
            value=avg_ctr,
            formatted=f"{avg_ctr:.2f}%",
            icon="📊",
        )
    )

    # 6. Top Channel
    if "channel" in current.columns and "conversions" in current.columns and not current.empty:
        ch_conv = current.groupby("channel")["conversions"].sum()
        top_ch = ch_conv.idxmax()
        results.append(
            KPIResult(
                label=f"Top Channel ({top_ch})",
                value=int(ch_conv.max()),
                formatted=_format_number(ch_conv.max()),
                icon="🏆",
            )
        )
    else:
        results.append(
            KPIResult(
                label="Top Channel",
                value=0,
                formatted="N/A",
                icon="🏆",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Generic KPIs (for unknown datasets)
# ---------------------------------------------------------------------------


def _generic_kpis(df: pd.DataFrame) -> list[KPIResult]:
    """Compute basic KPIs for any dataset."""
    results = [
        KPIResult(label="Total Rows", value=len(df), formatted=f"{len(df):,}", icon="📋"),
        KPIResult(
            label="Columns", value=len(df.columns), formatted=str(len(df.columns)), icon="📊"
        ),
    ]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in list(numeric_cols)[:4]:  # up to 4 numeric KPIs
        val = df[col].sum()
        results.append(
            KPIResult(
                label=f"Sum: {col}",
                value=val,
                formatted=_format_number(val, decimals=1),
                icon="📈",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

KPI_REGISTRY: dict[str, Callable[[pd.DataFrame], list[KPIResult]]] = {
    "ecommerce_orders": _ecommerce_kpis,
    "saas_subscriptions": _saas_kpis,
    "marketing_campaigns": _marketing_kpis,
}


def compute_kpis(df: pd.DataFrame, dataset_key: str) -> list[KPIResult]:
    """Compute KPIs appropriate for the given dataset.

    Args:
        df: Transformed DataFrame.
        dataset_key: Dataset identifier.

    Returns:
        List of ``KPIResult`` objects.
    """
    fn = KPI_REGISTRY.get(dataset_key, _generic_kpis)
    results = fn(df)
    logger.info(
        "KPIs computed",
        extra={"dataset": dataset_key, "kpi_count": len(results)},
    )
    return results
