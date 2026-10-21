"""Analytical summary generator.

Produces a concise narrative from computed metrics:
  1. Rule-based (always available): trend direction, biggest mover, anomalies
  2. LLM-generated (behind feature flag): grounded only on computed stats

The rule-based engine is deterministic and testable. The LLM path is
optional and gracefully degrades.
"""

from __future__ import annotations

import httpx
import numpy as np
import pandas as pd

from src.config import settings
from src.kpis import KPIResult
from src.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rule-based summary
# ---------------------------------------------------------------------------


def _trend_direction(df: pd.DataFrame, period_col: str, value_col: str) -> str:
    """Determine trend via linear regression slope."""
    if period_col not in df.columns or value_col not in df.columns:
        return "flat"

    agg = df.groupby(period_col)[value_col].sum().reset_index()
    if len(agg) < 3:
        return "insufficient data"

    x = np.arange(len(agg))
    y = agg[value_col].values.astype(float)

    if np.std(y) == 0:
        return "flat"

    slope = np.polyfit(x, y, 1)[0]
    mean_y = np.mean(y)

    # Normalize slope relative to the mean
    if mean_y == 0:
        return "flat"
    normalized = slope / abs(mean_y)

    if normalized > 0.02:
        return "upward"
    elif normalized < -0.02:
        return "downward"
    return "flat"


def _biggest_mover(
    df: pd.DataFrame,
    period_col: str,
    value_col: str,
    category_col: str,
) -> tuple[str, float] | None:
    """Find the category with the largest absolute change between first and second half."""
    if not all(c in df.columns for c in [period_col, value_col, category_col]):
        return None

    periods = sorted(df[period_col].unique())
    if len(periods) < 2:
        return None

    mid = len(periods) // 2
    first_periods = set(periods[:mid])
    second_periods = set(periods[mid:])

    first = df[df[period_col].isin(first_periods)].groupby(category_col)[value_col].sum()
    second = df[df[period_col].isin(second_periods)].groupby(category_col)[value_col].sum()

    changes = (second - first).fillna(0)
    if changes.empty:
        return None

    biggest = changes.abs().idxmax()
    return str(biggest), float(changes[biggest])


def _detect_anomalies(
    df: pd.DataFrame,
    period_col: str,
    value_col: str,
    sigma: float = 2.0,
) -> list[dict[str, str | float]]:
    """Flag periods where the metric exceeds ±sigma from rolling mean."""
    if period_col not in df.columns or value_col not in df.columns:
        return []

    agg = df.groupby(period_col)[value_col].sum().reset_index()
    if len(agg) < 5:
        return []

    anomalies = []
    for idx, row in agg.iterrows():
        # Evaluate point against surrounding window, excluding the point itself to prevent variance inflation
        window_indices = [i for i in range(max(0, idx - 2), min(len(agg), idx + 3)) if i != idx]
        local_pts = agg[value_col].iloc[window_indices]
        local_mean = local_pts.mean()
        local_std = local_pts.std()
        if pd.isna(local_std) or local_std == 0:
            local_std = agg[value_col].std()
        if local_std == 0 or pd.isna(local_std):
            continue

        z = abs(row[value_col] - local_mean) / local_std
        if z > sigma:
            direction = "spike" if row[value_col] > local_mean else "dip"
            anomalies.append(
                {
                    "period": str(row[period_col]),
                    "value": float(row[value_col]),
                    "direction": direction,
                    "z_score": round(float(z), 1),
                }
            )

    return anomalies


def generate_rule_based_summary(
    df: pd.DataFrame,
    kpis: list[KPIResult],
    dataset_key: str,
    period_col: str = "period_month",
    value_col: str = "revenue",
    category_col: str = "product_category",
) -> str:
    """Generate a concise rule-based analytical narrative.

    Args:
        df: Transformed DataFrame.
        kpis: Computed KPI results.
        dataset_key: Dataset identifier (for context).
        period_col: Period column for trend analysis.
        value_col: Primary metric column.
        category_col: Dimension for biggest-mover analysis.

    Returns:
        Markdown-formatted summary string.
    """
    # Adapt columns by dataset
    if dataset_key == "saas_subscriptions":
        value_col = "mrr"
        category_col = "plan"
    elif dataset_key == "marketing_campaigns":
        value_col = "conversions"
        category_col = "channel"

    sections = []

    # 1. Key trend
    trend = _trend_direction(df, period_col, value_col)
    trend_emoji = {"upward": "📈", "downward": "📉", "flat": "➡️"}.get(trend, "📊")
    primary_kpi = kpis[0] if kpis else None
    kpi_text = f" ({primary_kpi.label}: **{primary_kpi.formatted}**)" if primary_kpi else ""
    sections.append(
        f"**Key Trend:** {trend_emoji} The primary metric shows an **{trend}** trend"
        f" over the selected period{kpi_text}."
    )

    # 2. Biggest mover
    mover = _biggest_mover(df, period_col, value_col, category_col)
    if mover:
        name, change = mover
        direction = "increased" if change > 0 else "decreased"
        sections.append(
            f"**Biggest Mover:** *{name}* {direction} by **{abs(change):,.0f}** "
            f"({value_col}) compared to the prior half of the period."
        )

    # 3. Anomalies
    anomalies = _detect_anomalies(df, period_col, value_col)
    if anomalies:
        top_anomalies = sorted(anomalies, key=lambda a: a["z_score"], reverse=True)[:3]
        anomaly_lines = []
        for a in top_anomalies:
            anomaly_lines.append(
                f"  - **{a['period']}**: {a['direction']} " f"({a['z_score']}σ from rolling mean)"
            )
        sections.append("**Anomalies Detected:**\n" + "\n".join(anomaly_lines))
    else:
        sections.append("**Anomalies:** None detected — the data is within normal bounds.")

    # 4. KPI highlights
    kpi_highlights = []
    for kpi in kpis[:4]:
        if kpi.delta is not None:
            arrow = "↑" if kpi.delta > 0 else "↓" if kpi.delta < 0 else "→"
            kpi_highlights.append(
                f"  - {kpi.icon} {kpi.label}: **{kpi.formatted}** ({arrow} {abs(kpi.delta):.1f}%)"
            )
        else:
            kpi_highlights.append(f"  - {kpi.icon} {kpi.label}: **{kpi.formatted}**")
    if kpi_highlights:
        sections.append("**Key Metrics:**\n" + "\n".join(kpi_highlights))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# LLM-generated summary (behind feature flag)
# ---------------------------------------------------------------------------


def generate_llm_summary(
    rule_based_summary: str,
    kpis: list[KPIResult],
) -> str | None:
    """Call an OpenAI-compatible API to refine the summary.

    Only receives computed stats — never raw data rows.

    Args:
        rule_based_summary: The rule-based narrative (as context).
        kpis: Computed KPI results.

    Returns:
        LLM-generated summary, or None on failure.
    """
    if not settings.ENABLE_LLM_SUMMARY or not settings.LLM_API_KEY:
        return None

    kpi_text = "\n".join(f"- {k.label}: {k.formatted} (delta: {k.delta}%)" for k in kpis)

    prompt = (
        "You are a senior data analyst. Based on the following computed metrics "
        "and rule-based analysis, write a concise 3-4 sentence executive summary. "
        "Be specific with numbers. Do not speculate beyond what the data shows.\n\n"
        f"## Computed KPIs\n{kpi_text}\n\n"
        f"## Rule-Based Analysis\n{rule_based_summary}"
    )

    try:
        response = httpx.post(
            f"{settings.LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
            json={
                "model": settings.LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.3,
            },
            timeout=15,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning(f"LLM summary failed, falling back to rule-based: {exc}")
        return None
