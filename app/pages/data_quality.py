"""
RiskLens Analytics — Data Quality Center
All metrics, charts, and tables are derived entirely from the CSV loaded into
st.session_state.portfolio_df.  No hardcoded values anywhere.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from app.components.header import render_header
from src.preprocessing.quality_checker import (assess_data_health,
                                               calculate_psi)
from app.components.charts import (PRIMARY_BLUE, ACCENT_GOLD,
                                   apply_corporate_layout)

# Columns that should never appear in feature-level quality analysis
_META_COLS = {
    "applicant_id_hash", "application_date", "application_dow", "set"
}


def render_page():

    # ══════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════
    render_header(
        page_title="Data Quality Center",
        page_subtitle=("Portfolio Schema Integrity, Outliers, "
                       "and Population Stability Index (PSI)"),
    )

    # ══════════════════════════════════════════════════════
    # SESSION DATA — loaded by main.py from loan_book.csv
    # ══════════════════════════════════════════════════════
    if "portfolio_df" not in st.session_state:
        st.error(
            "Portfolio data not loaded. Please return to the Data Core page.")
        return

    df: pd.DataFrame = st.session_state.portfolio_df

    # Derive the feature list dynamically if not already in session state
    if "features" in st.session_state and st.session_state.features:
        features = [
            f for f in st.session_state.features
            if f in df.columns and f not in _META_COLS
        ]
    else:
        # Fallback: all non-meta columns
        features = [c for c in df.columns if c not in _META_COLS]

    # ══════════════════════════════════════════════════════
    # DATA HEALTH ASSESSMENT
    # ══════════════════════════════════════════════════════
    # Only pass numeric features — quantile/IQR in quality_checker
    # cannot handle string (large_string / pyarrow) columns
    numeric_features = [
        c for c in features
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    health = assess_data_health(df, numeric_features)

    # ══════════════════════════════════════════════════════
    # TOP SECTION: HEALTH SCORE GAUGE + MISSING REPORT
    # ══════════════════════════════════════════════════════
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Data Health Score")
        score = health["health_score"]

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
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
                        "tickcolor": PRIMARY_BLUE,
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
                    "steps": [
                        {
                            "range": [0, 50],
                            "color": "#FEE2E2"
                        },
                        {
                            "range": [50, 80],
                            "color": "#FEF3C7"
                        },
                        {
                            "range": [80, 100],
                            "color": "#D1FAE5"
                        },
                    ],
                },
            ))
        fig_gauge.update_layout(height=280,
                                margin=dict(l=20, r=20, t=40, b=20))
        apply_corporate_layout(fig_gauge)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.metric(
            label="Average Missingness",
            value=f"{health['average_missing_percentage']}%",
        )
        st.metric(
            label="Overall Outlier Rate",
            value=f"{health['overall_outlier_rate']}%",
        )

    with col2:
        st.markdown("### Missing Value Report")
        missing_df = health["missing_report"]
        st.dataframe(
            missing_df.style.format({"Missing Percentage": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )

    # ══════════════════════════════════════════════════════
    # OUTLIERS — fully data-driven from IQR bounds in health dict
    # ══════════════════════════════════════════════════════
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
            round(bounds["outlier_percentage"], 2),
        })

    if outliers_data:
        outlier_df = pd.DataFrame(outliers_data)
        st.dataframe(outlier_df, use_container_width=True, hide_index=True)
    else:
        st.info("No numeric columns available for outlier analysis.")

    # ══════════════════════════════════════════════════════
    # OUTLIER CHART — bar chart of outlier rates per variable
    # Built entirely from the outliers_data computed above
    # ══════════════════════════════════════════════════════
    if outliers_data:
        outlier_df_sorted = (pd.DataFrame(outliers_data).sort_values(
            "Outlier Rate %", ascending=False))

        fig_outlier = go.Figure(
            go.Bar(
                x=outlier_df_sorted["Variable"],
                y=outlier_df_sorted["Outlier Rate %"],
                marker_color=PRIMARY_BLUE,
                hovertemplate=("<b>%{x}</b><br>Outlier Rate: %{y:.2f}%"
                               "<br>Count: %{customdata:,}<extra></extra>"),
                customdata=outlier_df_sorted["Outlier Count"],
            ))
        fig_outlier.add_hline(
            y=outlier_df_sorted["Outlier Rate %"].mean(),
            line_dash="dash",
            line_color=ACCENT_GOLD,
            annotation_text=(
                f"Portfolio Avg "
                f"({outlier_df_sorted['Outlier Rate %'].mean():.2f}%)"),
            annotation_position="top right",
            annotation_font_size=10,
        )
        fig_outlier.update_layout(
            title="Outlier Rate (%) by Variable",
            xaxis_title="Variable",
            yaxis_title="Outlier Rate (%)",
            height=340,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        apply_corporate_layout(fig_outlier)
        st.plotly_chart(fig_outlier, use_container_width=True)

    # ══════════════════════════════════════════════════════
    # PSI — Population Stability Index Drift Monitor
    # ══════════════════════════════════════════════════════
    st.markdown("## Population Stability Index (PSI) Drift Monitor")
    st.info("PSI measures whether the current portfolio distribution "
            "has drifted away from the model training population.")

    # --- Detect the date column dynamically ---
    date_col = next(
        (c for c in ["application_date", "app_date", "timestamp"]
         if c in df.columns),
        None,
    )

    if date_col is None:
        st.warning(
            "No temporal column (e.g. 'application_date') found in the dataset."
        )
        return

    # Parse dates robustly — the CSV has mixed formats (ISO, DD/MM/YYYY, etc.)
    # infer_datetime_format was removed in pandas 2.0; use format="mixed" instead
    df[date_col] = pd.to_datetime(df[date_col],
                                  format="mixed",
                                  dayfirst=False,
                                  errors="coerce")

    valid_dates_df = df.dropna(subset=[date_col]).sort_values(by=date_col)

    if valid_dates_df.empty:
        st.error(
            "No valid dates found after parsing — cannot segment populations for PSI."
        )
        return

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

    # Only run PSI on numeric features that exist in both halves
    numeric_features = [
        f for f in features
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
        and f in baseline_df.columns and f in target_df.columns
    ]

    psi_records = []
    for col in numeric_features:
        psi_val, _ = calculate_psi(
            baseline_df[col].dropna().values,
            target_df[col].dropna().values,
        )
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

    if not psi_records:
        st.info("No numeric feature columns available to compute PSI.")
        return

    psi_df = pd.DataFrame(psi_records).sort_values(by="PSI Statistic",
                                                   ascending=False)

    # --- PSI Table ---
    st.dataframe(psi_df, use_container_width=True, hide_index=True)

    # --- PSI Bar Chart (fully data-driven) ---
    status_color_map = {
        "Stable": PRIMARY_BLUE,
        "Moderate Drift": ACCENT_GOLD,
        "Critical Drift": "#EF4444",
    }
    bar_colors = [
        status_color_map.get(s, PRIMARY_BLUE) for s in psi_df["Status"]
    ]

    fig_psi = go.Figure(
        go.Bar(
            x=psi_df["Variable"],
            y=psi_df["PSI Statistic"],
            marker_color=bar_colors,
            hovertemplate=("<b>%{x}</b><br>PSI: %{y:.4f}<extra></extra>"),
        ))
    for threshold, color, label in [
        (0.1, ACCENT_GOLD, "Moderate Drift (0.10)"),
        (0.25, "#EF4444", "Critical Drift (0.25)"),
    ]:
        fig_psi.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=color,
            line_width=1.5,
            annotation_text=label,
            annotation_position="top right",
            annotation_font_size=9,
            annotation_font_color=color,
        )
    fig_psi.update_layout(
        title="PSI by Variable — Population Stability Overview",
        xaxis_title="Variable",
        yaxis_title="PSI Statistic",
        height=360,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    apply_corporate_layout(fig_psi)
    st.plotly_chart(fig_psi, use_container_width=True)
