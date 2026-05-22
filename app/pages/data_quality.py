import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from app.components.header import render_header
from src.preprocessing.quality_checker import (assess_data_health,
                                               calculate_psi)
from app.components.charts import (PRIMARY_BLUE, ACCENT_GOLD,
                                   apply_corporate_layout)


def render_page():

    # ======================================================
    # HEADER
    # ======================================================
    render_header(page_title="Data Quality Center",
                  page_subtitle=("Portfolio Schema Integrity, Outliers, "
                                 "and Population Stability Index (PSI)"))

    # ======================================================
    # SESSION DATA
    # ======================================================
    df = st.session_state.portfolio_df
    features = st.session_state.features

    # ======================================================
    # DATA HEALTH
    # ======================================================
    health = assess_data_health(df, features)

    # ======================================================
    # TOP SECTION: METRICS & MISSING REPORT
    # ======================================================
    col1, col2 = st.columns([1, 2])

    # --- HEALTH SCORE PANEL ---
    with col1:
        st.markdown("### Data Health Score")
        score = health["health_score"]

        fig_gauge = go.Figure(
            go.Indicator(mode="gauge+number",
                         value=score,
                         domain={
                             "x": [0, 1],
                             "y": [0, 1]
                         },
                         title={
                             "text": "Health Rating",
                             "font": {
                                 "size": 16
                             }
                         },
                         gauge={
                             "axis": {
                                 "range": [0, 100],
                                 "tickwidth": 1,
                                 "tickcolor": PRIMARY_BLUE
                             },
                             "bar": {
                                 "color": PRIMARY_BLUE
                             },
                             "bgcolor":
                             "white",
                             "borderwidth":
                             2,
                             "bordercolor":
                             "#E2E8F0",
                             "steps": [{
                                 "range": [0, 50],
                                 "color": "#FEE2E2"
                             }, {
                                 "range": [50, 80],
                                 "color": "#FEF3C7"
                             }, {
                                 "range": [80, 100],
                                 "color": "#D1FAE5"
                             }]
                         }))

        fig_gauge.update_layout(height=280,
                                margin=dict(l=20, r=20, t=40, b=20))
        apply_corporate_layout(fig_gauge)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.metric(label="Average Missingness",
                  value=f"{health['average_missing_percentage']}%")
        st.metric(label="Overall Outlier Rate",
                  value=f"{health['overall_outlier_rate']}%")

    # --- MISSING REPORT PANEL ---
    with col2:
        st.markdown("### Missing Value Report")
        missing_df = health["missing_report"]

        st.dataframe(missing_df.style.format({"Missing Percentage":
                                              "{:.2f}%"}),
                     use_container_width=True,
                     hide_index=True)

    # ======================================================
    # OUTLIERS SECTION
    # ======================================================
    st.markdown("## Numeric Outlier Analysis")

    outliers_data = []
    for col, bounds in health["outliers_details"].items():
        outliers_data.append({
            "Variable":
            col,
            "Lower Bound":
            round(bounds["lower_bound"], 2),
            "Upper Bound":
            round(bounds["upper_bound"], 2),
            "Outlier Count":
            bounds["outlier_count"],
            "Outlier Rate %":
            round(bounds["outlier_percentage"], 2)
        })

    outlier_df = pd.DataFrame(outliers_data)
    st.dataframe(outlier_df, use_container_width=True, hide_index=True)

    # ======================================================
    # PSI SECTION
    # ======================================================
    st.markdown("## Population Stability Index (PSI) Drift Monitor")
    st.info("PSI measures whether the current portfolio distribution "
            "has drifted away from the model training population.")

    # --- DATE SAFETY AND PARSING ---
    # Safe lookup fallback check for variant dynamic timestamp column targets
    date_col = None
    for col in ["application_date", "app_date", "timestamp"]:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        st.warning(
            "Temporal index indicator column (e.g., 'application_date') missing."
        )
        return

    # Ensure native pandas datetime tracking
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Drop rows containing missing timestamps to maintain identical evaluation sizes
    valid_dates_df = df.dropna(subset=[date_col]).sort_values(by=date_col)

    if valid_dates_df.empty:
        st.error("No valid datetimes available to segment populations.")
        return

    # Determine midpoints safely from clean temporal distribution row index arrays
    midpoint_idx = len(valid_dates_df) // 2
    midpoint_date = valid_dates_df[date_col].iloc[midpoint_idx]

    col_l, col_r = st.columns(2)
    with col_l:
        st.success(
            f"Baseline Population:\n\nBefore {midpoint_date.strftime('%Y-%m-%d')}"
        )

    with col_r:
        st.warning(
            f"Target Population:\n\nOn/After {midpoint_date.strftime('%Y-%m-%d')}"
        )

    baseline_df = valid_dates_df[valid_dates_df[date_col] < midpoint_date]
    target_df = valid_dates_df[valid_dates_df[date_col] >= midpoint_date]

    # --- PSI CALCULATIONS LOOP ---
    psi_records = []
    for col in features:
        if col not in baseline_df.columns or col not in target_df.columns:
            continue

        psi_val, _ = calculate_psi(baseline_df[col].dropna().values,
                                   target_df[col].dropna().values)

        if psi_val < 0.1:
            status = "Stable"
        elif psi_val < 0.25:
            status = "Moderate Drift"
        else:
            status = "Critical Drift"

        psi_records.append({
            "Variable": col,
            "PSI Statistic": round(psi_val, 4),
            "Status": status
        })

    if psi_records:
        psi_df = pd.DataFrame(psi_records).sort_values(by="PSI Statistic",
                                                       ascending=False)
        st.dataframe(psi_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No matching numeric feature columns available to compute stability indexes."
        )
