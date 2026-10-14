"""
dashboard.py
------------
Optional Streamlit dashboard for the FreightTiger AI Shipping Cost Monitor.

Run with:
    streamlit run dashboard.py

Requirements: streamlit, pandas, numpy (all in requirements.txt)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FreightTiger AI Cost Monitor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
OUTPUT_CSV = PROJECT_ROOT / "output.csv"
DATA_DIR = PROJECT_ROOT / "data"
SHIPMENT_CSV = DATA_DIR / "shipment_records.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_output() -> pd.DataFrame | None:
    if not OUTPUT_CSV.exists():
        return None
    df = pd.read_csv(OUTPUT_CSV, parse_dates=["week_of"])
    return df


@st.cache_data(show_spinner=False)
def load_shipments() -> pd.DataFrame | None:
    if not SHIPMENT_CSV.exists():
        return None
    df = pd.read_csv(SHIPMENT_CSV, parse_dates=["shipment_date"])
    df["route"] = df["origin"].str.strip() + "-" + df["destination"].str.strip()
    return df


def run_pipeline_live():
    """Run the pipeline directly from the dashboard."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.pipeline import run_pipeline
    with st.spinner("Running analysis pipeline..."):
        metrics = run_pipeline(verbose=False)
    st.cache_data.clear()
    return metrics


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🚚 FreightTiger AI")
    st.markdown("**Shipping Cost Monitor**")
    st.divider()

    if st.button("🔄 Re-run Analysis", type="primary", use_container_width=True):
        metrics = run_pipeline_live()
        st.success(f"Done! Detected {metrics['anomalies_detected']} anomalies in {metrics['execution_time_s']:.1f}s")

    st.divider()
    st.markdown("### About")
    st.markdown(
        "Monitors freight costs per tonne-km across Indian logistics routes, "
        "flags anomalies using an 8-week rolling baseline and peer comparison, "
        "and generates AI-powered explanations via Groq (Llama 3.1)."
    )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = load_output()

if df is None:
    st.error("No output.csv found. Please run the pipeline first (click 'Re-run Analysis' in the sidebar).")
    st.stop()

# ---------------------------------------------------------------------------
# KPI Header
# ---------------------------------------------------------------------------

total_weeks = len(df)
total_anomalies = int((df["flagged"].notna() & (df["flagged"] != "")).sum())
justified = int((df["flagged"] == "No (justified)").sum())
unexplained = int((df["flagged"] == "Yes").sum())

st.title("FreightTiger AI — Shipping Cost Monitor")
st.markdown("*Powered by Groq (Llama 3.1 8B Instant) + deterministic Python analytics*")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Route-Weeks Analysed", f"{total_weeks:,}")
with col2:
    st.metric("Anomalies Detected", f"{total_anomalies}", delta=None)
with col3:
    st.metric("Justified by Evidence", f"{justified}", delta=None)
with col4:
    st.metric("Unexplained (Need Review)", f"{unexplained}", delta=f"+{unexplained}" if unexplained > 0 else None, delta_color="inverse")

st.divider()

# ---------------------------------------------------------------------------
# Route filter
# ---------------------------------------------------------------------------

routes = sorted(df["route"].unique())
selected_routes = st.multiselect(
    "Filter by route",
    options=routes,
    default=routes,
    placeholder="Select one or more routes...",
)

df_filtered = df[df["route"].isin(selected_routes)] if selected_routes else df

# ---------------------------------------------------------------------------
# Route-wise Cost Trend Chart
# ---------------------------------------------------------------------------

st.subheader("Route Cost Trends (Cost per Tonne-Km over time)")

import json

# Build chart data for each selected route
chart_data = df_filtered.pivot_table(
    index="week_of",
    columns="route",
    values="cost_per_tonne_km",
    aggfunc="mean",
)

st.line_chart(chart_data, height=400)

# ---------------------------------------------------------------------------
# Anomalies Table
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Detected Anomalies")

anomalies_df = df_filtered[df_filtered["flagged"].notna() & (df_filtered["flagged"] != "")].copy()

if anomalies_df.empty:
    st.info("No anomalies detected for the selected routes.")
else:
    # Color coding
    def color_flagged(val):
        if val == "Yes":
            return "background-color: #ffcccc; color: #cc0000; font-weight: bold"
        elif val == "No (justified)":
            return "background-color: #ccffcc; color: #006600; font-weight: bold"
        return ""

    display_df = anomalies_df[["route", "week_of", "cost_per_tonne_km",
                                "vs_own_history", "vs_similar_routes",
                                "flagged", "matched_note_id"]].copy()
    display_df["week_of"] = display_df["week_of"].dt.strftime("%Y-%m-%d")

    # pandas >= 2.1 uses .map(), older versions used .applymap()
    if hasattr(display_df.style, "map"):
        styled = display_df.style.map(color_flagged, subset=["flagged"])
    else:
        styled = display_df.style.applymap(color_flagged, subset=["flagged"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# AI Explanations
# ---------------------------------------------------------------------------

st.divider()
st.subheader("AI-Generated Explanations")

explanation_df = df_filtered[
    df_filtered["flagged"].notna() &
    (df_filtered["flagged"] != "") &
    df_filtered["reason"].notna()
].copy()

if explanation_df.empty:
    st.info("No explanations available.")
else:
    for _, row in explanation_df.iterrows():
        icon = "🔴" if row["flagged"] == "Yes" else "🟢"
        week_str = row["week_of"].strftime("%Y-%m-%d") if hasattr(row["week_of"], "strftime") else str(row["week_of"])

        with st.expander(f"{icon} **{row['route']}** — Week of {week_str}  |  {row['flagged']}"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Cost/Tonne-Km", f"₹{row['cost_per_tonne_km']:.2f}")
            with c2:
                st.markdown(f"**vs History:** {row['vs_own_history']}")
            with c3:
                st.markdown(f"**vs Peers:** {row['vs_similar_routes']}")

            if row["matched_note_id"] and str(row["matched_note_id"]) != "nan":
                st.info(f"**Matched Note:** {row['matched_note_id']}")

            st.markdown(f"**AI Explanation:** {row['reason']}")

# ---------------------------------------------------------------------------
# Full data table
# ---------------------------------------------------------------------------

st.divider()
with st.expander("View Full Dataset"):
    display_full = df_filtered.copy()
    display_full["week_of"] = display_full["week_of"].dt.strftime("%Y-%m-%d")
    st.dataframe(display_full, use_container_width=True, hide_index=True)

st.caption("FreightTiger AI Intern Case Study | Built with Python, Pandas, Groq API (Llama 3.1 8B)")
