import streamlit as st
from typing import List, Dict, Any


def render_kpi_card(label: str,
                    value: str,
                    delta_text: str = "",
                    delta_direction: str = "neutral"):

    if delta_direction == "up":
        delta_color = "#10B981"
        arrow = "▲"

    elif delta_direction == "down":
        delta_color = "#EF4444"
        arrow = "▼"

    else:
        delta_color = "#64748B"
        arrow = "•"

    with st.container():

        st.markdown("""
            <div class="kpi-card-wrapper">
            """,
                    unsafe_allow_html=True)

        st.caption(label.upper())

        st.markdown(f"""
            <div class="kpi-main-value">
                {value}
            </div>
            """,
                    unsafe_allow_html=True)

        st.markdown(f"""
            <div style="
                color:{delta_color};
                font-size:0.82rem;
                font-weight:600;
                margin-top:0.5rem;
            ">
                {arrow} {delta_text}
            </div>
            """,
                    unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


def render_kpi_row(cards: List[Dict[str, Any]]):

    cols = st.columns(len(cards))

    for idx, card in enumerate(cards):

        with cols[idx]:

            render_kpi_card(label=card.get("label", ""),
                            value=card.get("value", ""),
                            delta_text=card.get("delta_text", ""),
                            delta_direction=card.get("delta_direction",
                                                     "neutral"))
