"""
RiskLens Analytics — Feature Engineering Studio
Enterprise-grade WoE / IV Analysis for Credit Risk Scorecard Engineering
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple
import anthropic

# ──────────────────────────────────────────────────────────────
# Utility functions (from existing module)
# ──────────────────────────────────────────────────────────────


def compute_woe_iv_table(df: pd.DataFrame, col: str, target_col: str,
                         bins: list) -> pd.DataFrame:
    temp = pd.DataFrame({'val': df[col], 'target': df[target_col]}).dropna()
    if len(temp) == 0:
        return pd.DataFrame()
    temp['bin'] = pd.cut(temp['val'], bins=bins, include_lowest=True)
    total_goods = (temp['target'] == 0).sum()
    total_bads = (temp['target'] == 1).sum()
    if total_goods == 0 or total_bads == 0:
        return pd.DataFrame()
    grouped = temp.groupby('bin', observed=False).agg(
        count=('target', 'count'),
        bads=('target', lambda x: (x == 1).sum()),
        goods=('target', lambda x: (x == 0).sum())).reset_index()
    grouped['good_dist'] = (grouped['goods'] / total_goods).replace(0, 0.0001)
    grouped['bad_dist'] = (grouped['bads'] / total_bads).replace(0, 0.0001)
    grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
    grouped['iv'] = (grouped['good_dist'] -
                     grouped['bad_dist']) * grouped['woe']
    grouped['count_%'] = (grouped['count'] / len(temp)) * 100
    grouped['bad_rate_%'] = (grouped['bads'] /
                             grouped['count'].replace(0, np.nan)) * 100
    grouped['bin_label'] = grouped['bin'].astype(str)
    return grouped


def auto_bin_numeric(df: pd.DataFrame,
                     col: str,
                     target_col: str,
                     max_bins: int = 5) -> Tuple[pd.DataFrame, float, list]:
    temp_series = df[col].dropna()
    if len(temp_series.unique()) <= 1:
        return pd.DataFrame(), 0.0, []
    q = np.linspace(0, 1, max_bins + 1)
    bin_edges = np.unique(np.percentile(temp_series, q * 100))
    if len(bin_edges) < 3:
        bin_edges = np.linspace(temp_series.min(), temp_series.max(),
                                max_bins + 1)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    bin_table = compute_woe_iv_table(df, col, target_col, list(bin_edges))
    if bin_table.empty:
        return pd.DataFrame(), 0.0, []
    return bin_table, float(bin_table['iv'].sum()), list(bin_edges)


def compute_iv_summary(df: pd.DataFrame, numerical_cols: List[str],
                       target_col: str) -> pd.DataFrame:
    records = []
    for col in numerical_cols:
        if col != target_col and col in df.columns:
            _, iv_val, _ = auto_bin_numeric(df, col, target_col)
            if iv_val < 0.02: strength = "Useless"
            elif iv_val < 0.1: strength = "Weak"
            elif iv_val < 0.3: strength = "Medium"
            elif iv_val < 0.5: strength = "Strong"
            else: strength = "Suspicious"
            records.append({
                'Variable': col,
                'IV': round(iv_val, 4),
                'Predictive Power': strength
            })
    summary_df = pd.DataFrame(records)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            'IV', ascending=False).reset_index(drop=True)
    return summary_df


# ──────────────────────────────────────────────────────────────
# Styling helpers
# ──────────────────────────────────────────────────────────────

NAVY = "#0D1B3E"
NAVY2 = "#132347"
BLUE = "#1A3A6B"
GOLD = "#C9922A"
GOLD_L = "#E8B84B"
GRAY = "#F4F5F7"
GRAY2 = "#E8EAED"
TEXT = "#1C2B4A"
TEXT2 = "#4A5568"
WHITE = "#FFFFFF"
RED = "#B83232"
GREEN = "#1E7A4D"
AMBER = "#C97A1A"

STRENGTH_COLORS = {
    "Useless": ("#6B7280", "#F3F4F6"),
    "Weak": ("#92400E", "#FEF3C7"),
    "Medium": ("#1E40AF", "#DBEAFE"),
    "Strong": ("#065F46", "#D1FAE5"),
    "Suspicious": ("#991B1B", "#FEE2E2"),
}

SUITABILITY = {
    "Useless": "Exclude",
    "Weak": "Review",
    "Medium": "Include (conditional)",
    "Strong": "Recommended",
    "Suspicious": "Flag for leakage review",
}

PAGE_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    color: {TEXT};
}}

.risklens-header {{
    background: linear-gradient(135deg, {NAVY} 0%, {BLUE} 100%);
    padding: 2rem 2.5rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 4px solid {GOLD};
}}
.risklens-header h1 {{
    color: {WHITE};
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.01em;
}}
.risklens-header p {{
    color: rgba(255,255,255,0.65);
    font-size: 0.82rem;
    margin: 0;
    font-weight: 400;
}}

.kpi-card {{
    background: {WHITE};
    border: 1px solid {GRAY2};
    border-top: 3px solid {GOLD};
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 8px rgba(13,27,62,0.07);
}}
.kpi-card .kpi-label {{
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {TEXT2};
    margin-bottom: 0.5rem;
}}
.kpi-card .kpi-value {{
    font-size: 2rem;
    font-weight: 700;
    color: {NAVY};
    line-height: 1;
}}
.kpi-card .kpi-sub {{
    font-size: 0.73rem;
    color: {TEXT2};
    margin-top: 0.35rem;
}}

.kpi-warning {{
    border-top-color: {RED};
}}
.kpi-warning .kpi-value {{
    color: {RED};
}}

.section-header {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.8rem 0 0.9rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid {GRAY2};
}}
.section-header .section-icon {{
    width: 28px;
    height: 28px;
    background: {NAVY};
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
}}
.section-header h2 {{
    font-size: 1rem;
    font-weight: 700;
    color: {NAVY};
    margin: 0;
    letter-spacing: -0.01em;
}}
.section-header .section-badge {{
    margin-left: auto;
    font-size: 0.68rem;
    font-weight: 600;
    color: {GOLD};
    background: rgba(201,146,42,0.1);
    border: 1px solid rgba(201,146,42,0.25);
    border-radius: 20px;
    padding: 0.15rem 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

.strength-badge {{
    display: inline-block;
    padding: 0.18rem 0.65rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

.leakage-alert {{
    background: #FFF5F5;
    border: 1px solid #FCA5A5;
    border-left: 4px solid {RED};
    border-radius: 8px;
    padding: 1rem 1.3rem;
    margin: 0.8rem 0;
}}
.leakage-alert .alert-title {{
    font-size: 0.82rem;
    font-weight: 700;
    color: {RED};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}}
.leakage-alert .alert-body {{
    font-size: 0.82rem;
    color: #7F1D1D;
    line-height: 1.5;
}}

.insight-panel {{
    background: linear-gradient(135deg, {NAVY} 0%, #1E3A6E 100%);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
}}
.insight-panel::before {{
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 120px; height: 120px;
    background: rgba(201,146,42,0.06);
    border-radius: 50%;
    transform: translate(30px, -30px);
}}
.insight-panel .insight-header {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {GOLD_L};
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}
.insight-panel .insight-body {{
    font-size: 0.85rem;
    color: rgba(255,255,255,0.88);
    line-height: 1.65;
}}

.rec-card {{
    background: {WHITE};
    border: 1px solid {GRAY2};
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 1px 4px rgba(13,27,62,0.05);
}}
.rec-card .rec-type {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {GOLD};
    margin-bottom: 0.2rem;
}}
.rec-card .rec-title {{
    font-size: 0.85rem;
    font-weight: 600;
    color: {NAVY};
    margin-bottom: 0.25rem;
}}
.rec-card .rec-body {{
    font-size: 0.78rem;
    color: {TEXT2};
    line-height: 1.5;
}}

.corr-warning {{
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    border-left: 4px solid {AMBER};
    border-radius: 8px;
    padding: 0.8rem 1.1rem;
    font-size: 0.8rem;
    color: #78350F;
}}

.woe-table-container {{
    background: {WHITE};
    border: 1px solid {GRAY2};
    border-radius: 8px;
    overflow: hidden;
}}
</style>
"""


def render_section_header(icon: str, title: str, badge: str = ""):
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="section-header">
        <span class="section-icon">{icon}</span>
        <h2>{title}</h2>
        {badge_html}
    </div>
    """,
                unsafe_allow_html=True)


def render_kpi(label: str, value, sub: str = "", warning: bool = False):
    cls = "kpi-card kpi-warning" if warning else "kpi-card"
    st.markdown(f"""
    <div class="{cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """,
                unsafe_allow_html=True)


def strength_badge_html(strength: str) -> str:
    color, bg = STRENGTH_COLORS.get(strength, ("#6B7280", "#F3F4F6"))
    return f'<span class="strength-badge" style="color:{color};background:{bg};">{strength}</span>'


# ──────────────────────────────────────────────────────────────
# AI Insight generation (streaming)
# ──────────────────────────────────────────────────────────────


def generate_ai_insight(feature: str, bin_table: pd.DataFrame, iv: float,
                        strength: str) -> str:
    """Generate streaming AI analyst commentary for the selected feature."""
    woe_values = bin_table['woe'].tolist()
    bin_labels = bin_table['bin_label'].tolist()
    bad_rates = bin_table['bad_rate_%'].tolist()
    bad_rates_fmt = [f"{v:.1f}%" for v in bad_rates]
    is_monotonic = all(woe_values[i] <= woe_values[i+1] for i in range(len(woe_values)-1)) or \
                   all(woe_values[i] >= woe_values[i+1] for i in range(len(woe_values)-1))

    prompt = f"""You are a senior credit risk analyst at a tier-1 bank, reviewing a feature for scorecard development.

Feature: {feature}
Information Value (IV): {iv:.4f} — Predictive Power: {strength}
Bin boundaries and WoE values:
{chr(10).join(f"  Bin {b}: WoE = {w:.4f}, Bad Rate = {br}" for b, w, br in zip(bin_labels, woe_values, bad_rates_fmt))}
Monotonic WoE trend detected: {is_monotonic}

Write a concise, professional 3–4 sentence analytical commentary covering:
1. Monotonicity assessment and its implication for scorecard suitability
2. Risk trend observation (e.g., which segment is highest/lowest risk)
3. Underwriting or credit policy implication
4. Final scorecard inclusion recommendation

Use formal, institutional tone. Be specific. No bullet points. Plain paragraph only."""

    client = anthropic.Anthropic()
    full_text = ""
    with client.messages.stream(model="claude-sonnet-4-20250514",
                                max_tokens=350,
                                messages=[{
                                    "role": "user",
                                    "content": prompt
                                }]) as stream:
        for text in stream.text_stream:
            full_text += text
    return full_text


def generate_recommendations(feature: str, iv: float, strength: str,
                             bin_table: pd.DataFrame,
                             df: pd.DataFrame) -> List[dict]:
    recs = []
    series = df[feature].dropna()
    skew = series.skew()

    if abs(skew) > 1.5:
        recs.append({
            "type":
            "Transformation",
            "title":
            "Logarithmic Transformation",
            "body":
            f"The variable exhibits skewness of {skew:.2f}. Apply log(x+1) to normalise the distribution and improve model linearity."
        })

    missing_pct = df[feature].isna().mean() * 100
    if missing_pct > 2:
        recs.append({
            "type":
            "Missing Value Treatment",
            "title":
            "Missing Value Indicator",
            "body":
            f"{missing_pct:.1f}% of values are missing. Create a binary indicator (1 = missing) to capture potential informative missingness."
        })

    woe_vals = bin_table['woe'].tolist()
    is_monotonic = all(woe_vals[i] <= woe_vals[i+1] for i in range(len(woe_vals)-1)) or \
                   all(woe_vals[i] >= woe_vals[i+1] for i in range(len(woe_vals)-1))

    if not is_monotonic and strength in ("Medium", "Strong"):
        recs.append({
            "type":
            "Binning Strategy",
            "title":
            "Monotonic Binning Enforcement",
            "body":
            "Non-monotonic WoE trend detected. Apply monotonic binning constraints (isotonic regression or manual merge) to ensure scorecard compliance."
        })

    if strength == "Strong":
        recs.append({
            "type":
            "Scorecard Inclusion",
            "title":
            "Risk Banding for Scorecard",
            "body":
            "Feature demonstrates strong predictive power. Formalise risk bands into scorecard points using WoE encoding and log-odds calibration."
        })

    if iv < 0.02:
        recs.append({
            "type":
            "Feature Pruning",
            "title":
            "Candidate for Exclusion",
            "body":
            "IV below 0.02 threshold. Unless mandated by regulatory or business rationale, exclude from final scorecard to reduce complexity."
        })

    return recs[:4]  # cap at 4


# ──────────────────────────────────────────────────────────────
# Main page
# ──────────────────────────────────────────────────────────────


def render_feature_engineering_studio():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────
    st.markdown("""
    <div class="risklens-header">
        <h1>⚙ Feature Engineering Studio</h1>
        <p>RiskLens Analytics · Credit Risk Scorecard Platform · WoE / IV Analysis Engine</p>
    </div>
    """,
                unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────
    TARGET = "default_flag"

    if "portfolio_df" not in st.session_state:
        try:
            st.session_state.portfolio_df = pd.read_csv(
                "data/raw/loan_book.csv")
        except FileNotFoundError:
            st.error(
                "Dataset not found at `data/raw/loan_book.csv`. Please load the portfolio first."
            )
            return

    df: pd.DataFrame = st.session_state.portfolio_df

    if TARGET not in df.columns:
        st.error(f"Target column `{TARGET}` not found in dataset.")
        return

    numerical_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET
    ]

    # ── Compute IV summary (cached in session) ─────────────────
    if "iv_summary" not in st.session_state:
        with st.spinner("Computing Information Values across all features…"):
            st.session_state.iv_summary = compute_iv_summary(
                df, numerical_cols, TARGET)

    iv_df: pd.DataFrame = st.session_state.iv_summary

    # ── KPI Metrics ───────────────────────────────────────────
    render_section_header("📊", "Executive Portfolio Overview", "KPI SUMMARY")

    n_total = len(numerical_cols)
    n_strong = len(iv_df[iv_df['Predictive Power'] == 'Strong'])
    n_medium = len(iv_df[iv_df['Predictive Power'] == 'Medium'])
    n_leakage = len(iv_df[iv_df['Predictive Power'] == 'Suspicious'])
    default_rate = df[TARGET].mean() * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi("Numerical Features", n_total, f"{len(df):,} observations")
    with col2:
        render_kpi("Strong Predictors", n_strong, "IV > 0.30")
    with col3:
        render_kpi("Medium Predictors", n_medium, "IV 0.10–0.30")
    with col4:
        render_kpi("Portfolio Default Rate", f"{default_rate:.1f}%",
                   f"{int(df[TARGET].sum()):,} defaults")
    with col5:
        render_kpi("Leakage Flags",
                   n_leakage,
                   "IV > 0.50 — review required",
                   warning=(n_leakage > 0))

    # ── IV Ranking Table ──────────────────────────────────────
    render_section_header("📋", "Information Value Ranking",
                          "FEATURE SCREENING")

    # Enrich for display
    display_df = iv_df.copy()
    display_df['Rank'] = range(1, len(display_df) + 1)
    display_df['Scorecard Suitability'] = display_df['Predictive Power'].map(
        SUITABILITY)
    display_df = display_df[[
        'Rank', 'Variable', 'IV', 'Predictive Power', 'Scorecard Suitability'
    ]]

    # Search filter
    search = st.text_input("🔍  Filter features",
                           placeholder="Type a feature name…",
                           label_visibility="collapsed")
    if search:
        display_df = display_df[display_df['Variable'].str.contains(
            search, case=False)]

    # Style the dataframe
    def style_power(val):
        color, bg = STRENGTH_COLORS.get(val, ("#6B7280", "#F3F4F6"))
        return f"background-color: {bg}; color: {color}; font-weight: 600; font-size: 0.75rem; border-radius: 4px;"

    styled = (display_df.style.applymap(
        style_power, subset=['Predictive Power']).format({
            'IV': '{:.4f}'
        }).set_properties(**{
            'font-size': '0.82rem',
            'padding': '0.5rem 0.75rem',
        }).set_table_styles([
            {
                'selector':
                'th',
                'props': [
                    ('background-color', NAVY),
                    ('color', WHITE),
                    ('font-size', '0.73rem'),
                    ('text-transform', 'uppercase'),
                    ('letter-spacing', '0.06em'),
                    ('padding', '0.6rem 0.75rem'),
                ]
            },
            {
                'selector': 'tr:nth-child(even)',
                'props': [
                    ('background-color', GRAY),
                ]
            },
        ]).hide(axis='index'))
    st.dataframe(styled, use_container_width=True, height=340)

    # ── IV Bar Chart ──────────────────────────────────────────
    iv_chart = go.Figure()
    colors_bar = [
        STRENGTH_COLORS.get(p, ("#6B7280", "#F3F4F6"))[0]
        for p in iv_df['Predictive Power']
    ]
    iv_chart.add_trace(
        go.Bar(
            x=iv_df['Variable'],
            y=iv_df['IV'],
            marker_color=colors_bar,
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>IV: %{y:.4f}<extra></extra>",
        ))
    iv_chart.add_hline(y=0.3,
                       line_dash="dash",
                       line_color=GREEN,
                       line_width=1.2,
                       annotation_text="Strong threshold (0.30)",
                       annotation_position="top left",
                       annotation_font_size=10,
                       annotation_font_color=GREEN)
    iv_chart.add_hline(y=0.1,
                       line_dash="dot",
                       line_color=AMBER,
                       line_width=1.2,
                       annotation_text="Medium threshold (0.10)",
                       annotation_position="top left",
                       annotation_font_size=10,
                       annotation_font_color=AMBER)
    iv_chart.add_hline(y=0.5,
                       line_dash="dash",
                       line_color=RED,
                       line_width=1.2,
                       annotation_text="Leakage threshold (0.50)",
                       annotation_position="top left",
                       annotation_font_size=10,
                       annotation_font_color=RED)
    iv_chart.update_layout(
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        margin=dict(l=0, r=0, t=10, b=0),
        height=260,
        xaxis=dict(title="",
                   tickfont=dict(size=10, color=TEXT2),
                   gridcolor=GRAY2,
                   linecolor=GRAY2),
        yaxis=dict(title="Information Value",
                   tickfont=dict(size=10, color=TEXT2),
                   gridcolor=GRAY2,
                   linecolor=GRAY2,
                   zeroline=False),
        font=dict(family="IBM Plex Sans"),
    )
    st.plotly_chart(iv_chart, use_container_width=True)

    # ── WoE Workbench ─────────────────────────────────────────
    render_section_header("🔬", "WoE Analysis Workbench", "INTERACTIVE")

    available_features = iv_df['Variable'].tolist(
    ) if not iv_df.empty else numerical_cols
    selected_feature = st.selectbox("Select feature for deep-dive analysis",
                                    options=available_features,
                                    format_func=lambda x: x,
                                    label_visibility="visible")

    if selected_feature:
        bin_table, iv_val, _ = auto_bin_numeric(df, selected_feature, TARGET)
        strength = iv_df.loc[iv_df['Variable'] == selected_feature,
                             'Predictive Power'].values
        strength = strength[0] if len(strength) else "Unknown"

        if bin_table.empty:
            st.warning(
                "Insufficient data to compute WoE bins for this feature.")
        else:
            # ── Leakage warning ───────────────────────────────
            if iv_val > 0.5:
                st.markdown(f"""
                <div class="leakage-alert">
                    <div class="alert-title">⚠ Potential Feature Leakage Detected</div>
                    <div class="alert-body">
                        <strong>{selected_feature}</strong> has an IV of <strong>{iv_val:.4f}</strong>, 
                        which exceeds the 0.50 threshold. This may indicate institutionally-derived information, 
                        target leakage, or a near-perfect predictor. Review feature provenance carefully before 
                        final model inclusion. Consider data lineage documentation and out-of-time validation.
                    </div>
                </div>
                """,
                            unsafe_allow_html=True)

            # ── WoE table ─────────────────────────────────────
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.8rem; margin:0.6rem 0 0.5rem;">
                <span style="font-size:0.78rem; font-weight:600; color:{TEXT2}; text-transform:uppercase; letter-spacing:0.06em;">
                    {selected_feature}
                </span>
                {strength_badge_html(strength)}
                <span style="margin-left:auto; font-size:0.78rem; color:{TEXT2};">
                    IV = <strong style="color:{NAVY};">{iv_val:.4f}</strong>
                </span>
            </div>
            """,
                        unsafe_allow_html=True)

            display_woe = bin_table[[
                'bin_label', 'count', 'count_%', 'goods', 'bads', 'bad_rate_%',
                'woe', 'iv'
            ]].copy()
            display_woe.columns = [
                'Bin', 'Count', 'Count %', 'Goods', 'Bads', 'Bad Rate %',
                'WoE', 'IV Contribution'
            ]
            woe_styled = (display_woe.style.format({
                'Count %': '{:.1f}%',
                'Bad Rate %': '{:.1f}%',
                'WoE': '{:.4f}',
                'IV Contribution': '{:.4f}'
            }).background_gradient(
                subset=['WoE'
                        ], cmap='RdYlGn', vmin=-2, vmax=2).background_gradient(
                            subset=['Bad Rate %'],
                            cmap='Reds',
                            vmin=0,
                            vmax=100).set_properties(**{
                                'font-size': '0.8rem',
                                'padding': '0.4rem 0.7rem'
                            }).set_table_styles([
                                {
                                    'selector':
                                    'th',
                                    'props': [('background-color', NAVY),
                                              ('color', WHITE),
                                              ('font-size', '0.72rem'),
                                              ('text-transform', 'uppercase'),
                                              ('letter-spacing', '0.06em'),
                                              ('padding', '0.5rem 0.7rem')]
                                },
                            ]).hide(axis='index'))
            st.dataframe(woe_styled, use_container_width=True)

            # ── WoE Trend Chart ───────────────────────────────
            render_section_header("📈", "WoE Risk Trend",
                                  "MONOTONICITY ANALYSIS")

            woe_fig = go.Figure()
            woe_fig.add_trace(
                go.Bar(
                    x=bin_table['bin_label'],
                    y=bin_table['woe'],
                    name='WoE',
                    marker_color=[
                        BLUE if w >= 0 else RED for w in bin_table['woe']
                    ],
                    marker_line_width=0,
                    opacity=0.85,
                    hovertemplate="<b>%{x}</b><br>WoE: %{y:.4f}<extra></extra>",
                ))
            woe_fig.add_trace(
                go.Scatter(
                    x=bin_table['bin_label'],
                    y=bin_table['woe'],
                    name='Trend',
                    mode='lines+markers',
                    line=dict(color=GOLD, width=2.5),
                    marker=dict(color=GOLD,
                                size=7,
                                line=dict(color=WHITE, width=1.5)),
                    hoverinfo='skip',
                ))
            woe_fig.add_hline(y=0, line_color=GRAY2, line_width=1)
            woe_fig.update_layout(
                plot_bgcolor=WHITE,
                paper_bgcolor=WHITE,
                height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(title="Risk Band",
                           tickfont=dict(size=9, color=TEXT2),
                           gridcolor=GRAY2,
                           linecolor=GRAY2),
                yaxis=dict(title="Weight of Evidence",
                           tickfont=dict(size=9, color=TEXT2),
                           gridcolor=GRAY2,
                           linecolor=GRAY2,
                           zeroline=False),
                legend=dict(orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            x=0,
                            font=dict(size=9, color=TEXT2)),
                font=dict(family="IBM Plex Sans"),
                showlegend=True,
            )
            st.plotly_chart(woe_fig, use_container_width=True)

            # ── AI Insight Panel ──────────────────────────────
            render_section_header("🤖", "AI Risk Analyst Insight",
                                  "POWERED BY CLAUDE")

            insight_placeholder = st.empty()
            if st.button("Generate Analyst Commentary", type="primary"):
                with st.spinner("Analysing feature risk profile…"):
                    insight_text = generate_ai_insight(selected_feature,
                                                       bin_table, iv_val,
                                                       strength)
                    st.session_state[
                        f"insight_{selected_feature}"] = insight_text

            if f"insight_{selected_feature}" in st.session_state:
                insight_text = st.session_state[f"insight_{selected_feature}"]
                st.markdown(f"""
                <div class="insight-panel">
                    <div class="insight-header">
                        ◆ &nbsp;AI Analyst Commentary — {selected_feature}
                    </div>
                    <div class="insight-body">{insight_text}</div>
                </div>
                """,
                            unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:{GRAY};border:1px solid {GRAY2};border-radius:8px;
                            padding:1.1rem 1.4rem;font-size:0.8rem;color:{TEXT2};">
                    Click <strong>Generate Analyst Commentary</strong> to produce AI-assisted 
                    risk interpretation for this feature.
                </div>
                """,
                            unsafe_allow_html=True)

            # ── Feature Recommendations ───────────────────────
            render_section_header("💡", "Engineering Recommendations",
                                  "ADVISORY")

            recs = generate_recommendations(selected_feature, iv_val, strength,
                                            bin_table, df)
            if recs:
                rec_cols = st.columns(min(len(recs), 2))
                for i, rec in enumerate(recs):
                    with rec_cols[i % 2]:
                        st.markdown(f"""
                        <div class="rec-card">
                            <div class="rec-type">{rec['type']}</div>
                            <div class="rec-title">{rec['title']}</div>
                            <div class="rec-body">{rec['body']}</div>
                        </div>
                        """,
                                    unsafe_allow_html=True)
                        st.markdown("<div style='height:0.5rem'></div>",
                                    unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-type">Status</div>
                    <div class="rec-title">No engineering actions required</div>
                    <div class="rec-body">Feature meets standard scorecard engineering criteria. 
                    Proceed with standard WoE encoding.</div>
                </div>
                """,
                            unsafe_allow_html=True)

    # ── Correlation & Stability ───────────────────────────────
    render_section_header("🔗", "Correlation & Multicollinearity Review",
                          "STABILITY")

    strong_features = iv_df[iv_df['Predictive Power'].isin(
        ['Strong', 'Medium'])]['Variable'].tolist()

    if len(strong_features) >= 2:
        corr_df = df[strong_features].corr()

        # Heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns.tolist(),
            y=corr_df.columns.tolist(),
            colorscale=[[0, "#1A3A6B"], [0.5, WHITE], [1, "#B83232"]],
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_df.values, 2),
            texttemplate="%{text}",
            textfont={"size": 9},
            hovertemplate="<b>%{x} × %{y}</b><br>r = %{z:.3f}<extra></extra>",
            showscale=True,
            colorbar=dict(thickness=12, len=0.8, tickfont=dict(size=9))))
        fig_corr.update_layout(
            plot_bgcolor=WHITE,
            paper_bgcolor=WHITE,
            height=max(300,
                       len(strong_features) * 38),
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(tickfont=dict(size=9, color=TEXT2), side="bottom"),
            yaxis=dict(tickfont=dict(size=9, color=TEXT2)),
            font=dict(family="IBM Plex Sans"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Multicollinearity warnings
        warnings_found = False
        for i in range(len(strong_features)):
            for j in range(i + 1, len(strong_features)):
                r = abs(corr_df.iloc[i, j])
                if r > 0.70:
                    warnings_found = True
                    st.markdown(f"""
                    <div class="corr-warning">
                        ⚠ <strong>High Multicollinearity:</strong> 
                        <code>{strong_features[i]}</code> and <code>{strong_features[j]}</code> 
                        have |r| = {r:.3f}. Consider excluding one variable or applying PCA 
                        to prevent inflated standard errors in logistic regression.
                    </div>
                    <div style="height:0.4rem"></div>
                    """,
                                unsafe_allow_html=True)

        if not warnings_found:
            st.markdown(f"""
            <div style="background:#ECFDF5;border:1px solid #6EE7B7;border-left:4px solid {GREEN};
                        border-radius:8px;padding:0.75rem 1.1rem;font-size:0.8rem;color:#065F46;">
                ✓ No high multicollinearity detected among strong/medium predictors (|r| ≤ 0.70 for all pairs).
                Feature set is suitable for logistic regression without VIF concerns.
            </div>
            """,
                        unsafe_allow_html=True)
    else:
        st.info(
            "Insufficient strong/medium predictors for correlation analysis.")

    # ── Footer ────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid {GRAY2};
                font-size:0.72rem;color:{TEXT2};display:flex;justify-content:space-between;align-items:center;">
        <span>RiskLens Analytics · Feature Engineering Studio · {df.shape[0]:,} records · {df.shape[1]} variables</span>
        <span style="color:{GOLD};font-weight:600;">CONFIDENTIAL — INTERNAL USE ONLY</span>
    </div>
    """,
                unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    st.set_page_config(
        page_title="Feature Engineering Studio · RiskLens",
        page_icon="⚙",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_feature_engineering_studio()
