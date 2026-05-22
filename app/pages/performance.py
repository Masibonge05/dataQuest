import streamlit as streamlit_ctx
import plotly.graph_objects as go

from app.components.header import render_header
from app.components.cards import render_kpi_row
from app.components.charts import (apply_corporate_layout, PRIMARY_BLUE,
                                   ACCENT_GOLD, ALERT_RED)

from src.evaluation.metrics import calculate_model_performance


def render_page():
    # Explicitly alias to ensure no function shadowing can break session access
    st = streamlit_ctx

    # ==========================================================
    # HEADER
    # ==========================================================
    render_header(
        page_title="Model Performance Center",
        page_subtitle=("Model Discrimination Power, ROC, "
                       "Gini, and Kolmogorov-Smirnov (KS) Statistics"))

    # ==========================================================
    # SESSION DATA
    # ==========================================================
    if "portfolio_df" not in st.session_state:
        st.error("Portfolio dataframe missing from active session state.")
        return

    df = st.session_state.portfolio_df

    # ==========================================================
    # DYNAMIC TARGET & SCORE COLUMN DETECTION
    # ==========================================================
    target_col = None
    for col in ["default_flag", "default", "bad_flag"]:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        st.error(
            "Dataset must contain an explicit target indicator column (e.g., 'default_flag')."
        )
        return

    score_col = None
    for col in ["credit_score", "score", "scorecard_points"]:
        if col in df.columns:
            score_col = col
            break

    if score_col is None:
        st.error(
            "Calculated assessment score field is missing from runtime context."
        )
        return

    # ==========================================================
    # PERFORMANCE METRICS CALCULATION
    # ==========================================================
    perf = calculate_model_performance(df[target_col].values,
                                       df[score_col].values)

    # ==========================================================
    # KPI CARDS
    # ==========================================================
    kpis = [{
        "label":
        "Area Under Curve (AUC)",
        "value":
        f"{perf['auc']:.4f}",
        "delta_text":
        "Excellent discrimination"
        if perf['auc'] > 0.75 else "Acceptable performance",
        "delta_direction":
        "up"
    }, {
        "label": "Gini Coefficient",
        "value": f"{perf['gini']:.4f}",
        "delta_text": "Target: > 0.40",
        "delta_direction": "neutral"
    }, {
        "label": "KS Statistic",
        "value": f"{perf['ks_statistic']:.4f}",
        "delta_text": f"Cutoff at score {int(perf['ks_score_cutoff'])}",
        "delta_direction": "up"
    }]

    render_kpi_row(kpis)
    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================================
    # DISCRIMINATION METRICS VISUAL GRID (ROC & KS CURVES)
    # ==========================================================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="content-card"><h3>ROC Curve (Receiver Operating Characteristic)</h3></div>',
            unsafe_allow_html=True)

        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(x=perf["roc_curve"]["fpr"],
                       y=perf["roc_curve"]["tpr"],
                       mode="lines",
                       name=f"Scorecard Model (AUC = {perf['auc']:.3f})",
                       line=dict(color=PRIMARY_BLUE, width=3)))
        fig_roc.add_trace(
            go.Scatter(x=[0, 1],
                       y=[0, 1],
                       mode="lines",
                       name="Random Classifier",
                       line=dict(color="#94A3B8", width=1.5, dash="dash")))
        fig_roc.update_layout(xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate",
                              height=380,
                              margin=dict(l=20, r=20, t=30, b=20))
        apply_corporate_layout(fig_roc)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        st.markdown(
            '<div class="content-card"><h3>Kolmogorov-Smirnov (KS) Separation</h3></div>',
            unsafe_allow_html=True)

        ks_data = perf["ks_curve"]
        cutoff_score = perf["ks_score_cutoff"]

        fig_ks = go.Figure()
        fig_ks.add_trace(
            go.Scatter(x=ks_data["score"],
                       y=ks_data["cum_bads_rate"],
                       mode="lines",
                       name="Cum % Bads",
                       line=dict(color=ALERT_RED, width=3)))
        fig_ks.add_trace(
            go.Scatter(x=ks_data["score"],
                       y=ks_data["cum_goods_rate"],
                       mode="lines",
                       name="Cum % Goods",
                       line=dict(color=PRIMARY_BLUE, width=3)))

        row_at_cutoff = ks_data.iloc[(
            ks_data["score"] - cutoff_score).abs().argsort()[:1]].iloc[0]
        y0 = row_at_cutoff["cum_goods_rate"]
        y1 = row_at_cutoff["cum_bads_rate"]

        fig_ks.add_shape(type="line",
                         x0=cutoff_score,
                         y0=y0,
                         x1=cutoff_score,
                         y1=y1,
                         line=dict(color=ACCENT_GOLD, width=3, dash="dot"))
        fig_ks.add_annotation(x=cutoff_score,
                              y=(y0 + y1) / 2,
                              text=f"KS = {perf['ks_statistic']:.3f}",
                              showarrow=True,
                              arrowhead=1,
                              ax=50,
                              ay=0,
                              font=dict(color=PRIMARY_BLUE, size=11))
        fig_ks.update_layout(xaxis_title="Credit Score",
                             yaxis_title="Cumulative Probability",
                             height=380,
                             margin=dict(l=20, r=20, t=30, b=20))
        apply_corporate_layout(fig_ks)
        st.plotly_chart(fig_ks, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================================
    # ROW 2: CAP CURVE
    # ==========================================================
    with st.container():
        st.markdown(
            '<div class="content-card"><h3>Cumulative Accuracy Profile (CAP) Curve</h3></div>',
            unsafe_allow_html=True)

        cap_data = perf["cap_curve"]
        fig_cap = go.Figure()
        fig_cap.add_trace(
            go.Scatter(x=cap_data["pop_pct"],
                       y=cap_data["model_cap"],
                       mode="lines",
                       name="Scorecard Model",
                       line=dict(color=PRIMARY_BLUE, width=3)))
        fig_cap.add_trace(
            go.Scatter(x=cap_data["pop_pct"],
                       y=cap_data["perfect_cap"],
                       mode="lines",
                       name="Perfect Model",
                       line=dict(color=ACCENT_GOLD, width=2, dash="dash")))
        fig_cap.add_trace(
            go.Scatter(x=cap_data["pop_pct"],
                       y=cap_data["random_cap"],
                       mode="lines",
                       name="Random Model",
                       line=dict(color="#94A3B8", width=2, dash="dash")))
        fig_cap.update_layout(
            xaxis_title="Population Sorted by Risk (Highest Risk First)",
            yaxis_title="Defaults Captured (%)",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20))
        apply_corporate_layout(fig_cap)
        st.plotly_chart(fig_cap, use_container_width=True)
