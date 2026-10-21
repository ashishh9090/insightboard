"""Plotly chart builders for InsightBoard.

Each function returns a ``plotly.graph_objects.Figure`` — the caller
(app.py) renders it with ``st.plotly_chart``.

Design: charts accept DataFrames, not raw SQL, because transforms
have already been applied. This keeps the chart layer thin and testable.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

COLORS = [
    "#6366F1",  # indigo
    "#8B5CF6",  # violet
    "#EC4899",  # pink
    "#F59E0B",  # amber
    "#10B981",  # emerald
    "#3B82F6",  # blue
    "#F97316",  # orange
    "#14B8A6",  # teal
]

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color="#E2E8F0"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the InsightBoard dark theme to a figure."""
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.1)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.1)", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Time-series chart
# ---------------------------------------------------------------------------


def time_series(
    df: pd.DataFrame,
    period_col: str,
    value_col: str,
    breakdown_col: str | None = None,
    title: str = "Trend Over Time",
) -> go.Figure:
    """Line chart of a metric over time, optionally broken down by a dimension.

    Args:
        df: Transformed DataFrame.
        period_col: Column with period labels (e.g. ``period_month``).
        value_col: Numeric column to aggregate.
        breakdown_col: Optional dimension to group by.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    if breakdown_col and breakdown_col in df.columns:
        agg = df.groupby([period_col, breakdown_col])[value_col].sum().reset_index()
        fig = px.line(
            agg,
            x=period_col,
            y=value_col,
            color=breakdown_col,
            title=title,
            color_discrete_sequence=COLORS,
        )
    else:
        agg = df.groupby(period_col)[value_col].sum().reset_index()
        fig = px.line(
            agg,
            x=period_col,
            y=value_col,
            title=title,
            color_discrete_sequence=COLORS,
        )
        fig.update_traces(line=dict(width=3))

    fig.update_traces(mode="lines+markers", marker=dict(size=5))
    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# Top-N bar chart
# ---------------------------------------------------------------------------


def top_n_bar(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    n: int = 10,
    title: str = "Top Categories",
) -> go.Figure:
    """Horizontal bar chart of the top-N categories by a metric.

    Args:
        df: Transformed DataFrame.
        category_col: Categorical column.
        value_col: Numeric column to sum.
        n: Number of top items to show.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    agg = (
        df.groupby(category_col)[value_col]
        .sum()
        .nlargest(n)
        .reset_index()
        .sort_values(value_col, ascending=True)
    )
    fig = px.bar(
        agg,
        x=value_col,
        y=category_col,
        orientation="h",
        title=title,
        color=value_col,
        color_continuous_scale=["#312E81", "#6366F1", "#A78BFA"],
    )
    fig.update_coloraxes(showscale=False)
    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# Distribution histogram
# ---------------------------------------------------------------------------


def distribution(
    df: pd.DataFrame,
    value_col: str,
    title: str = "Distribution",
    nbins: int = 40,
) -> go.Figure:
    """Histogram showing the distribution of a numeric column.

    Args:
        df: Transformed DataFrame.
        value_col: Numeric column.
        title: Chart title.
        nbins: Number of bins.

    Returns:
        Plotly Figure.
    """
    fig = px.histogram(
        df,
        x=value_col,
        nbins=nbins,
        title=title,
        color_discrete_sequence=["#6366F1"],
    )
    fig.update_traces(marker_line_color="#4F46E5", marker_line_width=1)
    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# KPI sparkline (small inline chart for KPI cards)
# ---------------------------------------------------------------------------


def kpi_sparkline(
    df: pd.DataFrame,
    period_col: str,
    value_col: str,
    height: int = 60,
) -> go.Figure:
    """Minimal sparkline for embedding in KPI cards.

    Args:
        df: Transformed DataFrame.
        period_col: Period column.
        value_col: Numeric column.
        height: Chart height in pixels.

    Returns:
        Minimal Plotly Figure.
    """
    agg = df.groupby(period_col)[value_col].sum().reset_index()
    fig = go.Figure(
        go.Scatter(
            x=agg[period_col],
            y=agg[value_col],
            mode="lines",
            line=dict(color="#6366F1", width=2),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.1)",
        )
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig
