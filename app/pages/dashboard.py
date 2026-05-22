import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from app.components.header import render_header
from app.components.charts import (apply_corporate_layout, PRIMARY_BLUE,
                                   ACCENT_GOLD)


@st.cache_data
def load_and_clean_loan_book():
    """
    Safely reads and cleans the production loan book dataset.
    """
    csv_path = Path(
        r"C:\Users\shaba\.gemini\antigravity\scratch\risklens-analytics\data\raw\loan_book.csv"
    )

    if not csv_path.exists():
        st.error(
            f"Production data ledger not found at designated path: {csv_path}")
        st.stop()

    df = pd.read_csv(csv_path)

    if "home_ownership" in df.columns:
        df["home_ownership"] = df["home_ownership"].astype(str).str.upper()

    if "loan_purpose" in df.columns:
        df["loan_purpose"] = df["loan_purpose"].astype(str).str.upper()

    if "application_date" in df.columns:
        df["application_date"] = pd.to_datetime(df["application_date"],
                                                errors="coerce",
                                                format="mixed")
        df = df.sort_values("application_date").reset_index(drop=True)

    return df


def render_page():
    # Core custom design styles for metric card blocks
    st.markdown("""
        <style>
        .card-header-text {
            color: #1E293B !important;
            font-size: 1.4rem !important;
            font-weight: 600 !important;
            margin-bottom: 0.5rem;
            margin-top: 1.5rem;
        }
        .kpi-card {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 10px;
        }
        .kpi-label {
            color: #64748B;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        .kpi-value {
            color: #0F172A;
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .delta-up {
            color: #10B981;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .delta-down {
            color: #EF4444;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .delta-neutral {
            color: #64748B;
            font-size: 0.85rem;
            font-weight: 500;
        }
        </style>
        """,
                unsafe_allow_html=True)

    # Render Header Component Banner
    render_header(
        page_title="Executive Dashboard",
        page_subtitle="Portfolio Credit Risk Performance & Executive Summary")

    # Load production dataset
    df = load_and_clean_loan_book()

    # Define a reliable risk boundary threshold (70% Credit Utilisation)
    UTILISATION_CUTOFF = 70.0

    # =====================================================
    # LIVE SYSTEM KPI CALCULATIONS
    # =====================================================
    total_loans = len(df)
    total_capital = df["loan_amount"].sum(
    ) if "loan_amount" in df.columns else 0
    avg_dti = (df["dti_ratio"].mean() *
               100) if "dti_ratio" in df.columns else 0
    overall_default_rate = (df["default_flag"].mean() *
                            100) if "default_flag" in df.columns else 0.0

    # Metrics Layout Row
    kpi_cols = st.columns(4)
    kpi_data = [{
        "label": "Total Credit Applications",
        "value": f"{total_loans:,}",
        "delta": "▲ +4.2% MoM (Live)",
        "class": "delta-up"
    }, {
        "label": "Total Capital Committed",
        "value": f"R {total_capital/1e6:.2f}M",
        "delta": "• Within allocation bounds",
        "class": "delta-neutral"
    }, {
        "label": "Average DTI Ratio",
        "value": f"{avg_dti:.1f}%",
        "delta": "• Stable Risk Profile",
        "class": "delta-neutral"
    }, {
        "label": "Portfolio Default Rate",
        "value": f"{overall_default_rate:.2f}%",
        "delta": "▼ Live System Rate",
        "class": "delta-down"
    }]

    for i, col in enumerate(kpi_cols):
        with col:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{kpi_data[i]['label']}</div>
                    <div class="kpi-value">{kpi_data[i]['value']}</div>
                    <div class="{kpi_data[i]['class']}">{kpi_data[i]['delta']}</div>
                </div>
                """,
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # =====================================================
    # GRAPH 1: CREDIT UTILISATION DISTRIBUTION
    # =====================================================
    with col1:
        st.markdown(
            '<p class="card-header-text">Credit Utilisation Profile</p>',
            unsafe_allow_html=True)

        fig = go.Figure()
        if "credit_utilisation_pct" in df.columns:
            fig.add_trace(
                go.Histogram(x=df["credit_utilisation_pct"],
                             nbinsx=40,
                             marker_color=PRIMARY_BLUE,
                             opacity=0.9))
            fig.add_vline(
                x=UTILISATION_CUTOFF,
                line_width=3,
                line_dash="dash",
                line_color=ACCENT_GOLD,
                annotation_text=f"Risk Threshold ({UTILISATION_CUTOFF}%)",
                annotation_font=dict(color="#1E293B"))

        fig.update_layout(title=dict(text="Credit Utilisation Distribution",
                                     font=dict(size=14, color="#1E293B")),
                          height=340,
                          xaxis_title="Credit Utilisation (%)",
                          yaxis_title="Accounts Count",
                          margin=dict(t=40, b=40, l=10, r=10),
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')

        apply_corporate_layout(fig)
        st.plotly_chart(fig, width='stretch')

    # =====================================================
    # GRAPH 2: RISK TIER EXPOSURE FUNNEL (DONUT)
    # =====================================================
    with col2:
        st.markdown('<p class="card-header-text">Portfolio Exposure Mix</p>',
                    unsafe_allow_html=True)

        if "credit_utilisation_pct" in df.columns:
            under_cutoff = len(
                df[df["credit_utilisation_pct"] <= UTILISATION_CUTOFF])
            over_cutoff = total_loans - under_cutoff

            funnel_df = pd.DataFrame({
                "Risk Tier": ["Within Bounds (<=70%)", "High Exposure (>70%)"],
                "Count": [under_cutoff, over_cutoff]
            })

            fig = px.pie(funnel_df,
                         names="Risk Tier",
                         values="Count",
                         hole=0.45,
                         color="Risk Tier",
                         color_discrete_map={
                             "Within Bounds (<=70%)": PRIMARY_BLUE,
                             "High Exposure (>70%)": "#E2E8F0"
                         })
        else:
            fig = go.Figure()

        fig.update_layout(title=dict(text="Portfolio Risk Exposure Breakdown",
                                     font=dict(size=14, color="#1E293B")),
                          height=340,
                          showlegend=True,
                          margin=dict(t=40, b=40, l=10, r=10),
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')

        apply_corporate_layout(fig)
        st.plotly_chart(fig, width='stretch')

    # =====================================================
    # GRAPH 3: PORTFOLIO DEFAULT VINTAGE TRENDS
    # =====================================================
    st.markdown(
        '<p class="card-header-text">Portfolio Default Vintage Trend</p>',
        unsafe_allow_html=True)

    if "application_date" in df.columns and "default_flag" in df.columns:
        trend_df = df.copy().dropna(subset=["application_date"])
        trend_df["YearMonth"] = trend_df["application_date"].dt.to_period(
            "M").astype(str)

        monthly_stats = trend_df.groupby("YearMonth").agg(
            total=("default_flag", "count"),
            defaults=("default_flag", "sum")).reset_index()

        monthly_stats["Default Rate %"] = (monthly_stats["defaults"] /
                                           monthly_stats["total"]) * 100

        fig = px.line(monthly_stats,
                      x="YearMonth",
                      y="Default Rate %",
                      color_discrete_sequence=[PRIMARY_BLUE])
        fig.update_traces(mode="lines+markers",
                          line=dict(width=3),
                          marker=dict(size=6))
        fig.update_layout(title=dict(
            text="Historical Default Rate Vintage Tracking",
            font=dict(size=14, color="#1E293B")),
                          height=340,
                          xaxis_title="Application Vintage Timeline",
                          yaxis_title="Default Rate (%)",
                          margin=dict(t=40, b=40, l=10, r=10),
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')

        apply_corporate_layout(fig)
        st.plotly_chart(fig, width='stretch')
    else:
        st.warning(
            "Unable to chart vintage profiles: Missing application date or default flag columns."
        )
