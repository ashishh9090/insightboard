"""InsightBoard — Interactive Analytics Dashboard.

Main Streamlit application that wires together all modules:
  data_loader → validation → transforms → kpis → charts → summary

Run with: streamlit run app.py
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from datetime import UTC, datetime
from io import BytesIO

import pandas as pd
import streamlit as st

from src.charts import distribution, time_series, top_n_bar
from src.config import settings
from src.data_loader import list_sample_datasets, load_sample_dataset, load_uploaded_file
from src.db import SQLiteBackend, get_backend
from src.kpis import compute_kpis
from src.logging_setup import get_logger, metrics
from src.summary import generate_llm_summary, generate_rule_based_summary
from src.transforms import (
    detect_categorical_columns,
    detect_numeric_columns,
    transform_dataset,
)
from src.validation import QualityReport

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="InsightBoard — Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.1) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99,102,241,0.15);
    }
    .kpi-icon { font-size: 1.5rem; margin-bottom: 0.3rem; }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #E2E8F0;
    }
    .kpi-delta {
        font-size: 0.75rem;
        margin-top: 0.3rem;
    }
    .kpi-delta.positive { color: #10B981; }
    .kpi-delta.negative { color: #EF4444; }
    .kpi-delta.neutral { color: #94A3B8; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #E2E8F0;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(99,102,241,0.3);
    }

    /* Quality report */
    .quality-pass { color: #10B981; font-weight: 600; }
    .quality-warn { color: #F59E0B; font-weight: 600; }
    .quality-fail { color: #EF4444; font-weight: 600; }

    /* Summary box */
    .summary-box {
        background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(16,185,129,0.05) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 1.5rem;
        line-height: 1.7;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Metrics expander */
    .metrics-table td {
        padding: 0.3rem 0.8rem;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    """Initialize session state with defaults."""
    defaults = {
        "db": None,
        "current_dataset": None,
        "current_df": None,
        "quality_report": None,
        "app_start_time": time.perf_counter(),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


# ---------------------------------------------------------------------------
# Database singleton
# ---------------------------------------------------------------------------


def _get_db() -> SQLiteBackend:
    """Get or create the DB backend (cached in session state)."""
    if st.session_state.db is None:
        st.session_state.db = get_backend()
    return st.session_state.db


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------


def _hash_filters(**kwargs) -> str:
    """Create a hash for cache keys from filter parameters."""
    raw = str(sorted(kwargs.items()))
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _sync_query_params_to_state() -> None:
    """Read URL query params and apply to session state."""
    params = st.query_params
    if "role" in params:
        st.session_state["role"] = params["role"]
    if "dataset" in params:
        st.session_state["url_dataset"] = params["dataset"]
    if "granularity" in params:
        st.session_state["granularity"] = params["granularity"]


def _sync_state_to_query_params(**kwargs) -> None:
    """Write filter state to URL query params."""
    for key, val in kwargs.items():
        if val is not None and val != "" and val != []:
            if isinstance(val, list):
                st.query_params[key] = ",".join(str(v) for v in val)
            else:
                st.query_params[key] = str(val)


# ---------------------------------------------------------------------------
# Data loading UI
# ---------------------------------------------------------------------------


def _render_quality_report(report: QualityReport) -> None:
    """Display a friendly data quality report."""
    if not report.issues:
        st.success(f"✅ **All quality checks passed** — {report.summary_text()}")
        return

    summary_class = (
        "quality-fail"
        if report.has_errors
        else "quality-warn" if report.has_warnings else "quality-pass"
    )
    st.markdown(
        f'<span class="{summary_class}">{report.summary_text()}</span>', unsafe_allow_html=True
    )

    with st.expander("📋 Data Quality Details", expanded=report.has_errors):
        for issue in report.issues:
            icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(issue.severity, "⚪")
            col_text = f" (`{issue.column}`)" if issue.column else ""
            st.markdown(f"{icon} **{issue.check}**{col_text}: {issue.message}")


def _load_data_section() -> tuple[pd.DataFrame | None, str | None]:
    """Sidebar section for dataset selection and upload."""
    st.sidebar.markdown("### 📂 Data Source")

    samples = list_sample_datasets()
    source_type = st.sidebar.radio(
        "Choose source",
        ["Sample Dataset", "Upload File"],
        horizontal=True,
        label_visibility="collapsed",
    )

    db = _get_db()
    dataset_key = None
    df = None
    report = None

    if source_type == "Sample Dataset":
        # Build display options
        options = list(samples.keys())
        labels = [f"{samples[k]['label']} — {samples[k]['description']}" for k in options]

        # Check for URL param
        url_dataset = st.session_state.get("url_dataset")
        default_idx = 0
        if url_dataset and url_dataset in options:
            default_idx = options.index(url_dataset)

        selected_label = st.sidebar.selectbox(
            "Select dataset",
            labels,
            index=default_idx,
        )
        dataset_key = options[labels.index(selected_label)]

        if dataset_key != st.session_state.get("current_dataset"):
            with st.spinner(f"Loading {samples[dataset_key]['label']}..."):
                try:
                    df, report = load_sample_dataset(dataset_key, db)
                    st.session_state.current_dataset = dataset_key
                    st.session_state.current_df = df
                    st.session_state.quality_report = report
                except Exception as exc:
                    st.error(f"Failed to load dataset: {exc}")
                    metrics.increment("errors")
                    return None, None
        else:
            df = st.session_state.current_df
            report = st.session_state.quality_report

    else:
        uploaded = st.sidebar.file_uploader(
            "Upload CSV or XLSX",
            type=["csv", "xlsx", "xls"],
            help=f"Max {settings.MAX_UPLOAD_MB} MB",
        )
        if uploaded is not None:
            file_key = f"upload_{uploaded.name}_{uploaded.size}"
            if file_key != st.session_state.get("current_dataset"):
                with st.spinner("Processing upload..."):
                    try:
                        df, report = load_uploaded_file(uploaded, uploaded.name, db)
                        dataset_key = report.dataset_name
                        st.session_state.current_dataset = file_key
                        st.session_state.current_df = df
                        st.session_state.quality_report = report
                    except ValueError as exc:
                        st.error(f"❌ {exc}")
                        return None, None
                    except Exception as exc:
                        st.error(f"An error occurred while processing your file: {exc}")
                        metrics.increment("errors")
                        return None, None
            else:
                df = st.session_state.current_df
                report = st.session_state.quality_report
                dataset_key = report.dataset_name if report else "uploaded_data"

    if report:
        _render_quality_report(report)

    return df, dataset_key


# ---------------------------------------------------------------------------
# Filter panel
# ---------------------------------------------------------------------------


def _apply_filters(df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """Render filter controls and apply them to the DataFrame."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filters")

    filtered = df.copy()

    # Date filter
    date_col = "_parsed_date"
    if date_col in filtered.columns:
        date_vals = filtered[date_col].dropna()
        if not date_vals.empty:
            min_date = date_vals.min().date()
            max_date = date_vals.max().date()

            # Read from query params
            params = st.query_params
            default_start = min_date
            default_end = max_date
            if "date_from" in params:
                with contextlib.suppress(Exception):
                    default_start = pd.to_datetime(params["date_from"]).date()
            if "date_to" in params:
                with contextlib.suppress(Exception):
                    default_end = pd.to_datetime(params["date_to"]).date()

            date_range = st.sidebar.date_input(
                "Date range",
                value=(default_start, default_end),
                min_value=min_date,
                max_value=max_date,
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start, end = date_range
                filtered = filtered[
                    (filtered[date_col].dt.date >= start) & (filtered[date_col].dt.date <= end)
                ]
                _sync_state_to_query_params(date_from=str(start), date_to=str(end))

    # Categorical filters
    cat_cols = detect_categorical_columns(filtered)
    for col in cat_cols[:3]:  # Limit to 3 categorical filters
        unique_vals = sorted(filtered[col].dropna().unique().tolist())
        if len(unique_vals) < 2:
            continue

        # Read from query params
        default_selection = unique_vals
        param_key = col.lower().replace(" ", "_")
        if param_key in st.query_params:
            param_vals = st.query_params[param_key].split(",")
            default_selection = [v for v in param_vals if v in unique_vals] or unique_vals

        selected = st.sidebar.multiselect(
            f"Filter: {col}",
            options=unique_vals,
            default=default_selection,
        )
        if selected and len(selected) < len(unique_vals):
            filtered = filtered[filtered[col].isin(selected)]
            _sync_state_to_query_params(**{param_key: selected})

    # Numeric range filter (first numeric column that isn't a derived field)
    numeric_cols = detect_numeric_columns(filtered)
    filterable = [
        c for c in numeric_cols if c not in ["_parsed_date"] and not c.startswith("period_")
    ]
    if filterable:
        num_col = filterable[0]
        col_min = float(filtered[num_col].min())
        col_max = float(filtered[num_col].max())
        if col_min < col_max:
            num_range = st.sidebar.slider(
                f"Range: {num_col}",
                min_value=col_min,
                max_value=col_max,
                value=(col_min, col_max),
            )
            filtered = filtered[
                (filtered[num_col] >= num_range[0]) & (filtered[num_col] <= num_range[1])
            ]

    metrics.increment("filter_interactions")
    return filtered


# ---------------------------------------------------------------------------
# KPI rendering
# ---------------------------------------------------------------------------


def _render_kpis(kpis: list) -> None:
    """Render KPI cards in a responsive grid."""
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis, strict=False):
        delta_class = "neutral"
        delta_text = ""
        if kpi.delta is not None:
            delta_class = (
                "positive" if kpi.delta > 0 else "negative" if kpi.delta < 0 else "neutral"
            )
            delta_text = kpi.delta_label

        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon">{kpi.icon}</div>
                <div class="kpi-label">{kpi.label}</div>
                <div class="kpi-value">{kpi.formatted}</div>
                <div class="kpi-delta {delta_class}">{delta_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Export panel
# ---------------------------------------------------------------------------


def _render_export(df: pd.DataFrame, summary_text: str, dataset_key: str) -> None:
    """Download buttons for filtered data and summary."""
    st.markdown('<div class="section-header">📥 Export</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Filtered Data (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"insightboard_{dataset_key}_filtered.csv",
            mime="text/csv",
        )

    with col2:
        md_content = f"# InsightBoard Report — {dataset_key}\n\n"
        md_content += f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        md_content += f"**Rows:** {len(df):,}\n\n---\n\n"
        md_content += summary_text
        st.download_button(
            "⬇️ Download Summary (Markdown)",
            data=md_content,
            file_name=f"insightboard_{dataset_key}_summary.md",
            mime="text/markdown",
        )

    with col3:
        # SQL preview for analyst view
        if st.session_state.get("role") == "Analyst":
            sql_preview = f'SELECT * FROM "{dataset_key}" LIMIT 100;'
            st.code(sql_preview, language="sql")


# ---------------------------------------------------------------------------
# System metrics panel
# ---------------------------------------------------------------------------


def _render_system_metrics() -> None:
    """System Metrics expander showing observability data."""
    with st.expander("⚙️ System Metrics", expanded=False):
        snap = metrics.snapshot()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**⏱ Timers**")
            for label, data in snap.get("timers", {}).items():
                st.markdown(
                    f"- **{label}**: {data['last_ms']:.0f}ms last · "
                    f"{data['avg_ms']:.0f}ms avg · {data['count']} calls"
                )

            # App uptime
            uptime = time.perf_counter() - st.session_state.get(
                "app_start_time", time.perf_counter()
            )
            st.markdown(f"- **App session**: {uptime:.1f}s")

        with col2:
            st.markdown("**📊 Counters**")
            for label, count in snap.get("counters", {}).items():
                st.markdown(f"- **{label}**: {count}")

            st.markdown("**🕐 Timestamps**")
            for label, ts in snap.get("timestamps", {}).items():
                st.markdown(f"- **{label}**: {ts}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Application entry point."""
    t_start = time.perf_counter()

    _sync_query_params_to_state()

    # Header
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.5rem;">
            <span style="font-size: 2rem;">📊</span>
            <div>
                <h1 style="margin: 0; font-size: 1.8rem; font-weight: 700;
                    background: linear-gradient(135deg, #6366F1, #8B5CF6, #EC4899);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    InsightBoard
                </h1>
                <p style="margin: 0; color: #94A3B8; font-size: 0.85rem;">
                    Interactive analytics dashboard · Upload, explore, summarize
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Role toggle
    role_options = ["Executive", "Analyst"]
    default_role = st.session_state.get("role", "Executive")
    role_idx = role_options.index(default_role) if default_role in role_options else 0

    role = st.sidebar.selectbox(
        "👤 View Mode",
        role_options,
        index=role_idx,
    )
    st.session_state["role"] = role
    _sync_state_to_query_params(role=role)

    # Load data
    df, dataset_key = _load_data_section()

    if df is None or dataset_key is None:
        st.info("👈 Select a sample dataset or upload a file to get started.")
        return

    # Transform
    t_transform = time.perf_counter()
    transformed = transform_dataset(df, dataset_key)
    metrics.record_time("transform", (time.perf_counter() - t_transform) * 1000)

    # Apply filters
    filtered = _apply_filters(transformed, dataset_key)

    if filtered.empty:
        st.warning("No data matches the current filters. Try adjusting your selection.")
        return

    # Granularity toggle
    st.sidebar.markdown("---")
    granularity = st.sidebar.radio(
        "📅 Time Granularity",
        ["Month", "Week", "Day"],
        index=0,
        horizontal=True,
    )
    period_col = f"period_{granularity.lower()}"
    _sync_state_to_query_params(granularity=granularity.lower())

    # ── KPI Panel ──────────────────────────────────────────
    st.markdown(
        '<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True
    )
    t_kpi = time.perf_counter()
    kpis = compute_kpis(filtered, dataset_key)
    metrics.record_time("kpi_compute", (time.perf_counter() - t_kpi) * 1000)
    _render_kpis(kpis)

    # ── Trend Exploration (skip for Executive unless expanded) ──
    st.markdown('<div class="section-header">🔬 Trend Exploration</div>', unsafe_allow_html=True)

    # Determine value and category columns
    value_cols_map = {
        "ecommerce_orders": ("revenue", "product_category"),
        "saas_subscriptions": ("mrr", "plan"),
        "marketing_campaigns": ("conversions", "channel"),
    }
    value_col, cat_col = value_cols_map.get(
        dataset_key,
        (detect_numeric_columns(filtered)[0] if detect_numeric_columns(filtered) else None, None),
    )
    cat_cols = detect_categorical_columns(filtered)
    if not cat_col and cat_cols:
        cat_col = cat_cols[0]

    if value_col and period_col in filtered.columns:
        # Breakdown selector
        breakdown_options = ["None"] + cat_cols
        breakdown = st.selectbox(
            "Breakdown by",
            breakdown_options,
            index=0,
            label_visibility="collapsed",
        )
        breakdown_col = None if breakdown == "None" else breakdown

        tab1, tab2, tab3 = st.tabs(["📈 Trend", "📊 Top-N", "📉 Distribution"])

        with tab1:
            fig = time_series(
                filtered,
                period_col=period_col,
                value_col=value_col,
                breakdown_col=breakdown_col,
                title=f"{value_col.replace('_', ' ').title()} by {granularity}",
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if cat_col:
                top_n = st.slider("Show top N", 3, 20, 10)
                fig = top_n_bar(
                    filtered,
                    category_col=cat_col,
                    value_col=value_col,
                    n=top_n,
                    title=f"Top {top_n} by {value_col.replace('_', ' ').title()}",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns available for top-N analysis.")

        with tab3:
            fig = distribution(
                filtered,
                value_col=value_col,
                title=f"Distribution of {value_col.replace('_', ' ').title()}",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Analytical Summary ─────────────────────────────────
    st.markdown('<div class="section-header">📝 Analytical Summary</div>', unsafe_allow_html=True)
    summary = generate_rule_based_summary(
        filtered,
        kpis,
        dataset_key,
        period_col=period_col,
        value_col=value_col or "revenue",
        category_col=cat_col or "product_category",
    )

    # Try LLM summary if enabled
    llm_summary = generate_llm_summary(summary, kpis)
    if llm_summary:
        st.markdown(f'<div class="summary-box">{llm_summary}</div>', unsafe_allow_html=True)
        with st.expander("📋 Rule-based summary"):
            st.markdown(summary)
    else:
        st.markdown(f'<div class="summary-box">\n\n{summary}\n\n</div>', unsafe_allow_html=True)

    # ── Analyst-only sections ──────────────────────────────
    if role == "Analyst":
        st.markdown('<div class="section-header">🗃️ Raw Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(
            filtered.head(200),
            use_container_width=True,
            height=300,
        )

        # SQL preview
        st.markdown('<div class="section-header">🔎 SQL Preview</div>', unsafe_allow_html=True)
        st.code(
            f'SELECT * FROM "{dataset_key}" WHERE 1=1\n'
            f"-- {len(filtered):,} rows after filters\n"
            f"LIMIT 100;",
            language="sql",
        )

    # ── Export ─────────────────────────────────────────────
    _render_export(filtered, summary, dataset_key)

    # ── System Metrics ────────────────────────────────────
    total_time = (time.perf_counter() - t_start) * 1000
    metrics.record_time("page_render", total_time)
    _render_system_metrics()

    # Footer
    st.markdown(
        """
        <div style="text-align: center; color: #475569; font-size: 0.75rem; margin-top: 2rem; padding: 1rem;">
            InsightBoard · Built with Streamlit, Pandas & Plotly ·
            <a href="https://github.com/ashishh9090/insightboard" target="_blank" style="color: #6366F1;">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
