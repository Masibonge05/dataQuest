import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

from app.components.header import render_header

# ==========================================================
# BRAND COLORS
# ==========================================================
FNB_TURQUOISE = "#00A7B5"
FNB_ORANGE = "#F58220"
FNB_BLACK = "#111111"
WHITE = "#FFFFFF"

# ==========================================================
# DATA PATH
# ==========================================================
BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "data" / "raw" / "loan_book.csv"

EXCLUDED_COLUMNS = {
    "applicant_id_hash", "application_date", "application_dow", "set"
}

# ==========================================================
# PAGE CSS  (hero + ai-box only — no layout divs)
# ==========================================================
PAGE_CSS = f"""
<style>
.rl-hero {{
    background: linear-gradient(135deg, {FNB_TURQUOISE}, #007A87);
    padding: 1.8rem 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.10);
}}
.rl-hero h2 {{
    margin: 0 0 0.8rem 0;
    font-size: 1.7rem;
    font-weight: 700;
}}
.rl-hero-stats {{
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
    font-size: 0.95rem;
}}
.ai-box {{
    background: linear-gradient(135deg, {FNB_BLACK}, #1E293B);
    color: white;
    padding: 1.4rem 1.6rem;
    border-radius: 16px;
    border-left: 6px solid {FNB_ORANGE};
    line-height: 1.7;
}}
</style>
"""

CHART_LAYOUT = dict(paper_bgcolor=WHITE, plot_bgcolor=WHITE, height=420)


# ==========================================================
# DATA HELPERS
# ==========================================================
@st.cache_data(show_spinner=True)
def load_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"CSV file missing: {CSV_PATH}")
        st.stop()
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    return df


def detect_target(df):
    for c in ["default_flag", "default", "loan_default", "target"]:
        if c in df.columns:
            return c
    return None


def detect_features(df, target_col):
    usable = [
        c for c in df.columns if c not in EXCLUDED_COLUMNS and c != target_col
    ]
    numeric_cols = [c for c in usable if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [
        c for c in usable if pd.api.types.is_object_dtype(df[c])
    ]
    return numeric_cols, categorical_cols


# ==========================================================
# SIDEBAR BUILDER — called by main.py inside its sidebar block
# Returns (selected_feature, compare_feature, show_outliers)
# so main.py can pass them straight into render_page().
# ==========================================================
def render_sidebar_controls(numeric_cols, categorical_cols):
    st.markdown("---")
    st.markdown(
        "<p style='color:#CBD5E1;font-size:13px;font-weight:700;"
        "letter-spacing:0.5px;margin-bottom:8px'>EDA CONTROLS</p>",
        unsafe_allow_html=True,
    )
    all_features = numeric_cols + categorical_cols
    selected_feature = st.selectbox("Select Feature",
                                    all_features,
                                    key="eda_selected_feature")
    compare_feature = st.selectbox("Compare Against",
                                   numeric_cols,
                                   key="eda_compare_feature")
    show_outliers = st.toggle("Show Outliers",
                              value=True,
                              key="eda_show_outliers")
    return selected_feature, compare_feature, show_outliers


# ==========================================================
# GEMINI COMMENTARY
# ==========================================================
def generate_ai_commentary(feature, stats_text):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ Gemini API key not configured. Add GEMINI_API_KEY to your .env file."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        You are a senior banking risk analyst.
        Generate executive-level credit risk commentary.

        Selected Feature: {feature}
        Statistics: {stats_text}

        Cover: underwriting implications, risk trends, borrower behavior,
        business impact, and portfolio implications.
        Be professional and enterprise-grade.
        """
        return model.generate_content(prompt).text
    except Exception as e:
        return f"AI Commentary Error: {e}"


# ==========================================================
# MAIN RENDER  — called from main.py as:
#
#   with st.sidebar:
#       sel, cmp, out = eda_lab.render_sidebar_controls(num, cat)
#   eda_lab.render_page(sel, cmp, out)
#
# render_page() does NOT touch st.sidebar at all.
# ==========================================================
def render_page(selected_feature, compare_feature, show_outliers):

    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    df = load_data()
    target_col = detect_target(df)
    numeric_cols, categorical_cols = detect_features(df, target_col)

    render_header(
        page_title="Interactive EDA Laboratory",
        page_subtitle=
        "Enterprise Credit Risk Intelligence & Portfolio Analytics",
    )

    # ── Hero ───────────────────────────────────────────────
    total_records = len(df)
    default_rate = df[target_col].mean() * 100 if target_col else 0.0

    st.markdown(f"""
    <div class="rl-hero">
        <h2>RiskLens Enterprise Analytics</h2>
        <div class="rl-hero-stats">
            <span><strong>Portfolio Size:</strong> {total_records:,} Accounts</span>
            <span><strong>Default Rate:</strong> {default_rate:.2f}%</span>
            <span><strong>Target Variable:</strong> {target_col}</span>
            <span><strong>Status:</strong> Active &amp; Operational</span>
        </div>
    </div>
    """,
                unsafe_allow_html=True)

    # ── KPIs ───────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Average Income",
                  f"R{df['annual_income'].mean():,.0f}",
                  help="Mean borrower income")
    with k2:
        st.metric("Average Loan",
                  f"R{df['loan_amount'].mean():,.0f}",
                  help="Portfolio exposure")
    with k3:
        st.metric("Avg Interest Rate",
                  f"{df['interest_rate'].mean():.2f}%",
                  help="Average pricing level")
    with k4:
        st.metric("Total Exposure",
                  f"R{df['loan_amount'].sum():,.0f}",
                  help="Total lending portfolio")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Diagnostics ────────────────────────────────
    st.subheader("Feature Diagnostics")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Missing Values", f"{df[selected_feature].isna().sum():,}")
    with d2:
        st.metric("Unique Values", f"{df[selected_feature].nunique():,}")
    with d3:
        if pd.api.types.is_numeric_dtype(df[selected_feature]):
            st.metric("Skewness", f"{df[selected_feature].skew():.2f}")
        else:
            st.metric("Feature Type", "Categorical")
    with d4:
        if pd.api.types.is_numeric_dtype(df[selected_feature]) and target_col:
            corr = df[selected_feature].corr(df[target_col])
            st.metric("Correlation w/ Target", f"{corr:.3f}")
        else:
            st.metric("Correlation", "N/A")

    st.divider()

    # ── Distribution + Risk Segmentation ──────────────────
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Distribution Analysis")
            if selected_feature in numeric_cols:
                hist_df = df.copy()
                if not show_outliers:
                    q1 = hist_df[selected_feature].quantile(0.01)
                    q99 = hist_df[selected_feature].quantile(0.99)
                    hist_df = hist_df[hist_df[selected_feature].between(
                        q1, q99)]
                fig = px.histogram(hist_df,
                                   x=selected_feature,
                                   nbins=40,
                                   marginal="box",
                                   color_discrete_sequence=[FNB_TURQUOISE])
            else:
                count_df = df[selected_feature].value_counts().reset_index()
                count_df.columns = [selected_feature, "count"]
                fig = px.bar(count_df,
                             x=selected_feature,
                             y="count",
                             color_discrete_sequence=[FNB_TURQUOISE])
            fig.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("Risk Segmentation")
            if selected_feature in numeric_cols and target_col:
                risk_df = df.copy()
                risk_df["feature_bin"] = (pd.qcut(
                    risk_df[selected_feature], q=5,
                    duplicates="drop").astype(str))
                grouped = (risk_df.groupby("feature_bin")
                           [target_col].mean().reset_index())
                grouped["default_rate_pct"] = grouped[target_col] * 100
                fig = px.bar(
                    grouped,
                    x="feature_bin",
                    y="default_rate_pct",
                    color="default_rate_pct",
                    color_continuous_scale=["#DCEEFF", FNB_ORANGE],
                    labels={
                        "feature_bin": "Feature Bin",
                        "default_rate_pct": "Default Rate (%)"
                    },
                )
                fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "Risk segmentation is available for numerical variables.")

    # ── Bivariate ──────────────────────────────────────────
    if selected_feature in numeric_cols and compare_feature in numeric_cols:
        with st.container(border=True):
            st.subheader("Bivariate Relationship Analysis")
            scatter_df = df.copy()
            if target_col:
                scatter_df["Risk"] = scatter_df[target_col].map({
                    0: "Good",
                    1: "Defaulted"
                })
                fig = px.scatter(scatter_df,
                                 x=selected_feature,
                                 y=compare_feature,
                                 color="Risk",
                                 opacity=0.65,
                                 color_discrete_map={
                                     "Good": FNB_TURQUOISE,
                                     "Defaulted": FNB_ORANGE
                                 })
            else:
                fig = px.scatter(scatter_df,
                                 x=selected_feature,
                                 y=compare_feature,
                                 color_discrete_sequence=[FNB_TURQUOISE])
            fig.update_layout(paper_bgcolor=WHITE,
                              plot_bgcolor=WHITE,
                              height=500)
            st.plotly_chart(fig, use_container_width=True)

    # ── Correlation Matrix ─────────────────────────────────
    with st.container(border=True):
        st.subheader("Correlation Intelligence Matrix")
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix,
                        text_auto=".2f",
                        aspect="auto",
                        color_continuous_scale=[[0, "#EFF6FF"],
                                                [0.5, "#FFFFFF"],
                                                [1, FNB_TURQUOISE]])
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, height=750)
        st.plotly_chart(fig, use_container_width=True)

    # ── Data Quality ───────────────────────────────────────
    with st.container(border=True):
        st.subheader("Enterprise Data Quality Monitoring")
        total_cells = df.shape[0] * df.shape[1]
        missing = int(df.isna().sum().sum())
        duplicates = int(df.duplicated().sum())
        completeness = 100 - (missing / total_cells * 100)
        integrity = 100 - (duplicates / len(df) * 100)
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric("Missing Values", f"{missing:,}")
        with q2:
            st.metric("Duplicate Rows", f"{duplicates:,}")
        with q3:
            st.metric("Completeness", f"{completeness:.2f}%")
        with q4:
            st.metric("Integrity Score", f"{integrity:.2f}%")

    # ── AI Commentary ──────────────────────────────────────
    st.subheader("AI Executive Commentary")
    mean_val = (f"{df[selected_feature].mean():.4f}"
                if selected_feature in numeric_cols else "N/A (categorical)")
    stats_text = (f"Mean: {mean_val} | "
                  f"Missing: {df[selected_feature].isna().sum()} | "
                  f"Unique Values: {df[selected_feature].nunique()}")
    ai_text = generate_ai_commentary(selected_feature, stats_text)
    st.markdown(f'<div class="ai-box">{ai_text}</div>', unsafe_allow_html=True)

    # ── Export ─────────────────────────────────────────────
    st.download_button(
        label="⬇ Download Portfolio Dataset",
        data=df.to_csv(index=False),
        file_name="risklens_portfolio_export.csv",
        mime="text/csv",
    )
