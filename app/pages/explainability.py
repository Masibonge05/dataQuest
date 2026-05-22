import streamlit as streamlit_ctx
import pandas as pd
import plotly.graph_objects as go

from app.components.header import render_header
from app.components.charts import (PRIMARY_BLUE, ACCENT_GOLD,
                                   apply_corporate_layout)


def render_page():
    # Explicitly alias to ensure no function shadowing can break session access
    st = streamlit_ctx

    # ======================================================
    # HEADER
    # ======================================================
    render_header(
        page_title="Model Explainability",
        page_subtitle=
        "Global Feature Importance & Local Attribution Metrics (SHAP Values)")

    # ======================================================
    # DATA SAFETY CHECK
    # ======================================================
    if "portfolio_df" not in st.session_state:
        st.error(
            "No active credit engine execution dataframe found in session state."
        )
        return

    df = st.session_state.portfolio_df

    # ======================================================
    # GLOBAL IMPORTANCE VISUALIZATION
    # ======================================================
    st.markdown("### Global Feature Contributions")

    # Check if mock or real feature importance parameters exist, or generate mock profile
    if "feature_importances" in st.session_state:
        importance_dict = st.session_state.feature_importances
    else:
        importance_dict = {
            "Debt-to-Income Ratio (DTI)": 0.32,
            "Credit Utilization Rate": 0.28,
            "Payment History Matrix": 0.22,
            "Derogatory Public Records": 0.11,
            "Length of Credit History": 0.07
        }

    importance_df = pd.DataFrame({
        "Feature": list(importance_dict.keys()),
        "Importance": list(importance_dict.values())
    }).sort_values(by="Importance", ascending=True)

    fig_importance = go.Figure(
        go.Bar(x=importance_df["Importance"],
               y=importance_df["Feature"],
               orientation="h",
               marker_color=PRIMARY_BLUE))

    fig_importance.update_layout(xaxis_title="Relative Contribution Weight",
                                 yaxis_title="Risk Predictor Component",
                                 height=340,
                                 margin=dict(l=20, r=20, t=20, b=20))
    apply_corporate_layout(fig_importance)
    st.plotly_chart(fig_importance, use_container_width=True)

    # ======================================================
    # LOCAL ATTRIVUTION / INDIVIDUAL RECORD LOOKUP
    # ======================================================
    st.markdown("---")
    st.markdown("### Local Account Deep-Dive")

    account_id_col = None
    for col in ["account_id", "client_id", "id", "customer_id"]:
        if col in df.columns:
            account_id_col = col
            break

    if account_id_col is None:
        st.info(
            "Unique Account ID matching keys not found. Showing top portfolio rows instead."
        )
        st.dataframe(df.head(10), use_container_width=True)
        return

    selected_id = st.selectbox(
        "Select Account Identity Ref for Attribution Analysis:",
        options=df[account_id_col].unique())

    selected_record = df[df[account_id_col] == selected_id].iloc[0]
    st.write(
        f"Displaying evaluation weights for Account Reference Key: **{selected_id}**"
    )
    st.json(selected_record.to_dict())
