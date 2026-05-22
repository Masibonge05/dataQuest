"""
RiskLens Analytics — Feature Engineering Studio
All analytics derived from real loan_book.csv data loaded into st.session_state.portfolio_df.
No hardcoded data or placeholder text anywhere.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai

from app.components.header import render_header
from app.components.charts import apply_corporate_layout, PRIMARY_BLUE, ACCENT_GOLD
from src.feature_engineering.binning import auto_bin_numeric, compute_iv_summary

# ──────────────────────────────────────────────────────────────
# Columns to always exclude from feature analysis
# ──────────────────────────────────────────────────────────────
_EXCLUDE_COLS = {
    "applicant_id_hash",
    "application_date",
    "application_dow",
    "branch_code_id",
    "set",
    "default_flag",
    "default",
    "email_domain_type",
    "home_ownership",
    "region",
    "loan_purpose",
    "phone_verified",
}

# ──────────────────────────────────────────────────────────────
# Professional FNB Corporate Design Tokens
# ──────────────────────────────────────────────────────────────
FNB_TURQUOISE = "#00A3C4"
FNB_GOLD = "#FFB800"
FNB_DARK_TEXT = "#0F172A"
FNB_NAVY = "#0A2540"
WHITE = "#FFFFFF"
GRAY_BG = "#F8FAFC"
BORDER_GRAY = "#E2E8F0"
TEXT_MUTED = "#64748B"

# Alert and Category Token Colors
CRITICAL_RED = "#EF4444"
COMPLIANT_GREEN = "#10B981"
WARNING_AMBER = "#F59E0B"

STRENGTH_META = {
    "Useless": {
        "color": "#64748B",
        "bg": "#F1F5F9",
        "suit": "Exclude"
    },
    "Weak": {
        "color": "#B45309",
        "bg": "#FEF3C7",
        "suit": "Review"
    },
    "Medium": {
        "color": "#0369A1",
        "bg": "#E0F2FE",
        "suit": "Include (conditional)"
    },
    "Strong": {
        "color": "#047857",
        "bg": "#D1FAE5",
        "suit": "Recommended"
    },
    "Suspicious": {
        "color": "#B91C1C",
        "bg": "#FEE2E2",
        "suit": "Flag — leakage review"
    },
}

PAGE_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
section[data-testid="stMain"] * {{ font-family: 'IBM Plex Sans', sans-serif !important; }}

.kpi-row {{ display:flex; gap:1rem; margin-bottom:1.5rem; flex-wrap:wrap; }}
.kpi-card {{
    flex:1; min-width:180px;
    background:{WHITE}; border:1px solid {BORDER_GRAY};
    border-top:4px solid {FNB_TURQUOISE}; border-radius:8px;
    padding:1.1rem 1.3rem;
    box-shadow:0 1px 3px rgba(15,23,42,0.05);
}}
.kpi-card.warn {{ border-top-color:{CRITICAL_RED}; }}
.kpi-label {{
    font-size:0.7rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.08em;
    color:{TEXT_MUTED}; margin-bottom:0.4rem;
}}
.kpi-value {{ font-size:1.8rem; font-weight:700; color:{FNB_DARK_TEXT}; line-height:1; }}
.kpi-card.warn .kpi-value {{ color:{CRITICAL_RED}; }}
.kpi-sub {{ font-size:0.75rem; color:{TEXT_MUTED}; margin-top:0.3rem; }}

.sec-hdr {{
    display:flex; align-items:center; gap:0.6rem;
    padding-bottom:0.5rem; border-bottom:2px solid {BORDER_GRAY};
    margin:1.8rem 0 1rem;
}}
.sec-hdr h3 {{
    font-size:1.1rem; font-weight:700; color:{FNB_NAVY};
    margin:0; letter-spacing:-0.01em;
}}
.sec-badge {{
    margin-left:auto; font-size:0.68rem; font-weight:700;
    color:{FNB_TURQUOISE}; background:rgba(0,163,196,0.08);
    border:1px solid rgba(0,163,196,0.25);
    border-radius:20px; padding:0.15rem 0.6rem;
    text-transform:uppercase; letter-spacing:0.07em;
}}

.s-badge {{
    display:inline-block; padding:0.15rem 0.6rem;
    border-radius:20px; font-size:0.72rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.05em;
}}

.feat-meta {{
    display:flex; align-items:center; gap:0.8rem;
    background:{GRAY_BG}; border:1px solid {BORDER_GRAY};
    border-radius:8px; padding:0.8rem 1.1rem;
    margin-bottom:0.8rem; font-size:0.82rem;
}}
.feat-name {{ font-weight:700; color:{FNB_NAVY}; font-size:0.95rem; }}
.feat-iv {{ margin-left:auto; color:{TEXT_MUTED}; }}
.feat-iv strong {{ color:{FNB_DARK_TEXT}; }}

.leakage-alert {{
    background:#FFF5F5; border:1px solid #FCA5A5;
    border-left:4px solid {CRITICAL_RED}; border-radius:8px;
    padding:1rem 1.3rem; margin:0.8rem 0;
}}
.al-title {{
    font-size:0.8rem; font-weight:700; color:{CRITICAL_RED};
    text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.3rem;
}}
.al-body {{ font-size:0.82rem; color:#7F1D1D; line-height:1.55; }}

.ai-panel {{
    background:linear-gradient(135deg, {FNB_NAVY} 0%, #153E66 100%);
    border-left:5px solid {FNB_GOLD};
    border-radius:8px; padding:1.4rem 1.6rem;
    margin-bottom:1rem;
}}
.ai-hdr {{
    font-size:0.72rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:{FNB_GOLD}; margin-bottom:0.6rem;
}}
.ai-body {{ font-size:0.86rem; color:rgba(255,255,255,0.92); line-height:1.65; }}

.rec-card {{
    background:{WHITE}; border:1px solid {BORDER_GRAY}; border-radius:8px;
    padding:0.9rem 1.1rem; box-shadow:0 1px 2px rgba(0,0,0,0.02);
    margin-bottom:0.6rem;
}}
.rec-type {{
    font-size:0.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.08em; color:{FNB_TURQUOISE}; margin-bottom:0.2rem;
}}
.rec-title {{ font-size:0.88rem; font-weight:700; color:{FNB_DARK_TEXT}; margin-bottom:0.25rem; }}
.rec-body  {{ font-size:0.8rem; color:{TEXT_MUTED}; line-height:1.5; }}
</style>
"""

# ──────────────────────────────────────────────────────────────
# Data Computations & Analytics
# ──────────────────────────────────────────────────────────────


def _derive_numeric_features(df: pd.DataFrame, target_col: str) -> list:
    candidates = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in _EXCLUDE_COLS and c != target_col
    ]
    valid = []
    for c in candidates:
        non_null = df[c].dropna()
        if len(non_null) == 0:
            continue
        top_freq = non_null.value_counts(normalize=True).iloc[0]
        if top_freq < 0.99:
            valid.append(c)
    return valid


def _feature_stats(df: pd.DataFrame, feature: str, target_col: str) -> dict:
    series = df[feature].dropna()
    defaults_per_bin = df.groupby(pd.qcut(df[feature], q=5, duplicates="drop"),
                                  observed=False)[target_col].mean()
    return {
        "n": len(series),
        "missing_n": int(df[feature].isna().sum()),
        "missing_pct": df[feature].isna().mean() * 100,
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "skew": series.skew(),
        "p5": series.quantile(0.05),
        "p95": series.quantile(0.95),
        "min": series.min(),
        "max": series.max(),
        "overall_default_rate": df[target_col].mean() * 100,
        "default_rate_by_quintile": defaults_per_bin.to_dict(),
    }


def _generate_ai_insight(feature: str, bin_table: pd.DataFrame, iv: float,
                         strength: str, stats: dict) -> str:
    try:
        _secrets_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        _secrets_key = None
    api_key = _secrets_key or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return "Analyst commentary unavailable — GEMINI_API_KEY is not configured in environment secrets."

    woe_vals = bin_table["woe"].tolist()
    bin_lbls = bin_table["bin_label"].tolist()
    bad_rates = bin_table["bad_rate_%"].tolist()
    counts = bin_table["count"].tolist()

    mono = (all(woe_vals[i] <= woe_vals[i + 1]
                for i in range(len(woe_vals) - 1))
            or all(woe_vals[i] >= woe_vals[i + 1]
                   for i in range(len(woe_vals) - 1)))

    max_bad_idx = int(np.argmax(bad_rates))
    min_bad_idx = int(np.argmin(bad_rates))

    prompt = f"""You are a senior banking credit risk analyst inspecting model weights.
Review this empirical feature extracted from our active consumer loan portfolio composed of {stats['n']:,} observation records.

FEATURE CAPTION: {feature}
REAL EMPIRICAL STATISTICS:
  - Mean Value: {stats['mean']:.4f}
  - Median Parameter: {stats['median']:.4f}
  - Tail Deviation: {stats['std']:.4f}
  - Skew Intensity: {stats['skew']:.4f}
  - Boundary (P5/P95): {stats['p5']:.4f} / {stats['p95']:.4f}
  - Missing Ledger Count: {stats['missing_n']:,} ({stats['missing_pct']:.2f}%)
  - Base Portfolio Default Rate: {stats['overall_default_rate']:.2f}%

WEIGHT OF EVIDENCE (WoE) METRICS:
  - Total Calculated Information Value (IV): {iv:.4f}
  - Assigned Class Power: {strength}
  - Strictly Monotonic Trend Flag: {mono}
  - Account Bin Structure:
""" + "\n".join(
        f"    Bin Index {i+1} [{bin_lbls[i]}]: Computed WoE={woe_vals[i]:.4f}, Default Ratio={bad_rates[i]:.2f}%, Volume Group={counts[i]:,}"
        for i in range(len(bin_lbls))) + f"""

Maximum Risk Segment Group: Bin Range Index {max_bad_idx+1} [{bin_lbls[max_bad_idx]}] yielding a default rate of {bad_rates[max_bad_idx]:.2f}%
Minimum Risk Segment Group: Bin Range Index {min_bad_idx+1} [{bin_lbls[min_bad_idx]}] yielding a default rate of {bad_rates[min_bad_idx]:.2f}%

Draft exactly 4 sentences of high-level credit risk commentary. Highlight if the trend complies with banking scorecard rules (monotonic consistency). Point directly to the highest risk band stats. State if the distributions require adjustments, and conclude with an exact include/exclude verdict based on empirical IV parameters. Do not add labels, headers, or bullet points."""

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower(
        ) or "ResourceExhausted" in err:
            return "Analyst commentary temporarily unavailable — System Google AI free tier rate limit constraints reached."
        return f"Analyst commentary error connection timeout: {err}"


def _recommendations(feature: str, iv: float, strength: str,
                     bin_table: pd.DataFrame, stats: dict) -> list:
    recs = []
    skew = stats["skew"]
    missing_pct = stats["missing_pct"]
    p5 = stats["p5"]
    p95 = stats["p95"]

    if abs(skew) > 1.5:
        direction = "right" if skew > 0 else "left"
        recs.append({
            "type":
            "Transformation Strategy",
            "title":
            f"Logarithmic Structural Transform Recommended",
            "body":
            f"The variable features high {direction}-skewness ({skew:.2f}) spanning a range of {p5:.2f} to {p95:.2f}. Transform via log(x+1) to align closer to log-odds linearity constraints required for credit scorecards."
        })

    if missing_pct > 2.0:
        recs.append({
            "type":
            "Missingness Correction",
            "title":
            "Establish Structural Missing Flag Indicator",
            "body":
            f"We detected {stats['missing_n']:,} rows ({missing_pct:.2f}%) with null items. Designate a unique binary weight indicator attribute to verify if unpopulated cells represent structural hidden risks."
        })

    woe_vals = bin_table["woe"].tolist()
    mono = (all(woe_vals[i] <= woe_vals[i + 1]
                for i in range(len(woe_vals) - 1))
            or all(woe_vals[i] >= woe_vals[i + 1]
                   for i in range(len(woe_vals) - 1)))

    if not mono and strength in ("Medium", "Strong"):
        recs.append({
            "type":
            "Bin Boundary Optimisation",
            "title":
            "Enforce Monotonic Trend Sequencing",
            "body":
            f"The raw metrics across the bins display a non-monotonic pattern. Apply monotonic pooling or adjacent-bin merging routines to establish clear logical underwriting risk progression."
        })

    if strength == "Strong":
        bad_rates = bin_table["bad_rate_%"].tolist()
        spread = max(bad_rates) - min(bad_rates)
        recs.append({
            "type":
            "Scorecard Entry Allocation",
            "title":
            "Approved Candidate for Point Mapping Assignment",
            "body":
            f"With a solid predictive IV score of {iv:.4f} and default delta gap spread reaching {spread:.1f} percentage points across risk buckets, this column is prioritized for scaling logic integration."
        })

    if iv < 0.02:
        recs.append({
            "type":
            "Dimensionality Pruning",
            "title":
            "Exclude Feature From Production Modeling Pipeline",
            "body":
            f"The Information Value coefficient ({iv:.4f}) sits entirely under the standard 0.02 credit model minimum benchmark threshold. Drop this feature to minimize noise profiles."
        })

    return recs[:4]


def _sec(title: str, badge: str = ""):
    badge_html = f'<span class="sec-badge">{badge}</span>' if badge else ""
    st.markdown(f'<div class="sec-hdr"><h3>{title}</h3>{badge_html}</div>',
                unsafe_allow_html=True)


def _badge(strength: str) -> str:
    m = STRENGTH_META.get(strength, {"color": "#64748B", "bg": "#F1F5F9"})
    return f'<span class="s-badge" style="color:{m["color"]}; background:{m["bg"]};">{strength}</span>'


def _iv_strength(iv: float) -> str:
    """Enforces absolute alignment with STRENGTH_META dictionary tokens."""
    if iv < 0.02: return "Useless"
    if iv < 0.1: return "Weak"
    if iv < 0.3: return "Medium"
    if iv < 0.5: return "Strong"
    return "Suspicious"


def _kpi(label: str, value, sub: str = "", warn: bool = False) -> str:
    cls = "kpi-card warn" if warn else "kpi-card"
    return f'<div class="{cls}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>'


# ──────────────────────────────────────────────────────────────
# Main Page Entry Point
# ──────────────────────────────────────────────────────────────


def render_page():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    render_header(
        page_title="Feature Engineering Studio",
        page_subtitle=
        "Weight of Evidence (WoE) Binning & Information Value (IV) Diagnostics Portfolio Summary Matrix",
    )

    # ── Load Core Data Framework ───────────────────────────
    if "portfolio_df" not in st.session_state:
        st.error(
            "Production portfolio data ledger not loaded inside framework memory pipeline. Return to the Data Core page."
        )
        st.stop()

    df: pd.DataFrame = st.session_state.portfolio_df

    # ── Verify Target Metric Columns ──────────────────────
    target_col = next(
        (c for c in ["default_flag", "default"] if c in df.columns), None)
    if target_col is None:
        st.error(
            "Target classification variable framework marker column missing from core loan_book file structure."
        )
        st.stop()

    # ── Map Numerical Features Dynamically ─────────────────
    if "features" in st.session_state and st.session_state.features:
        features = [
            f for f in st.session_state.features
            if f in df.columns and f not in _EXCLUDE_COLS and f != target_col
            and pd.api.types.is_numeric_dtype(df[f])
        ]
    else:
        features = _derive_numeric_features(df, target_col)

    if not features:
        st.error(
            "Zero standard numeric indicators discovered within current portfolio array schema options."
        )
        st.stop()

    # ── Initialize Framework IV Matrix Processing ───────────
    if "iv_summary" not in st.session_state:
        with st.spinner(
                f"Evaluating statistical Information Value limits for {len(features)} active columns..."
        ):
            raw_iv = compute_iv_summary(df, features, target_col)
            col_map = {
                c: "IV"
                for c in raw_iv.columns
                if c.lower() in ("iv", "information value (iv)",
                                 "information value")
            }
            if col_map:
                raw_iv = raw_iv.rename(columns=col_map)
            st.session_state.iv_summary = raw_iv

    iv_df: pd.DataFrame = st.session_state.iv_summary

    if "IV" in iv_df.columns:
        iv_df["Predictive Power"] = iv_df["IV"].apply(_iv_strength)

    # ══════════════════════════════════════════════════════════
    # 1. EXECUTIVE METRIC CARDS — All Dynamic From Data
    # ══════════════════════════════════════════════════════════
    _sec("Executive Portfolio Overview", "KPI SUMMARY")

    n_strong = len(iv_df[iv_df["Predictive Power"] == "Strong"])
    n_medium = len(iv_df[iv_df["Predictive Power"] == "Medium"])
    n_leakage = len(iv_df[iv_df["Predictive Power"] == "Suspicious"])
    def_rate = df[target_col].mean() * 100
    n_defaults = int(df[target_col].sum())
    missing_avg = df[features].isna().mean().mean() * 100

    st.markdown(
        '<div class="kpi-row">' + _kpi("Numerical Indicators", len(features),
                                       f"{len(df):,} accounts logged") +
        _kpi("Strong Predictors", n_strong, "IV Range Matrix > 0.30") +
        _kpi("Medium Predictors", n_medium, "IV Limit 0.10 – 0.30") +
        _kpi("Base Default Rate", f"{def_rate:.2f}%",
             f"{n_defaults:,} default events") +
        _kpi("Mean Indicator Missingness", f"{missing_avg:.2f}%",
             "Across array data blocks") + _kpi("Leakage Warnings",
                                                n_leakage,
                                                "IV Risk Metric > 0.50 flag",
                                                warn=(n_leakage > 0)) +
        "</div>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════
    # 2. DATA-DRIVEN INFOMATION VALUE SORT RANKING MATRIX
    # ══════════════════════════════════════════════════════════
    _sec("Information Value Ranking Matrix", "FEATURE SCREENING")

    disp = iv_df.copy()
    disp = disp.sort_values(by="IV", ascending=False).reset_index(drop=True)
    disp.insert(0, "Rank", range(1, len(disp) + 1))
    disp["Scorecard Suitability"] = disp["Predictive Power"].map(
        lambda p: STRENGTH_META.get(p, {}).get("suit", "—"))
    disp = disp[[
        c for c in [
            "Rank", "Variable", "IV", "Predictive Power",
            "Scorecard Suitability"
        ] if c in disp.columns
    ]]

    def _style_power(val):
        m = STRENGTH_META.get(val, {"color": "#64748B", "bg": "#F1F5F9"})
        return f"background-color:{m['bg']}; color:{m['color']}; font-weight:700; font-size:0.75rem; border-radius:4px;"

    styled_iv = (disp.style.map(_style_power,
                                subset=["Predictive Power"]).format({
                                    "IV":
                                    "{:.4f}"
                                }).set_properties(**{
                                    "font-size": "0.82rem",
                                    "padding": "0.5rem 0.8rem"
                                }).set_table_styles([{
                                    "selector":
                                    "th",
                                    "props": [("background-color", FNB_NAVY),
                                              ("color", WHITE),
                                              ("font-size", "0.75rem"),
                                              ("text-transform", "uppercase"),
                                              ("letter-spacing", "0.05em"),
                                              ("padding", "0.6rem 0.8rem")]
                                }, {
                                    "selector":
                                    "tr:nth-child(even)",
                                    "props": [("background-color", GRAY_BG)]
                                }]).hide(axis="index"))

    st.dataframe(styled_iv, width='stretch', height=260)

    # ── Live Visual IV Histogram Deployment ──────────────────
    bar_colors = [
        STRENGTH_META.get(p, {"color": "#64748B"})["color"]
        for p in iv_df["Predictive Power"]
    ]
    fig_iv = go.Figure(
        go.Bar(
            x=iv_df["Variable"],
            y=iv_df["IV"],
            marker_color=bar_colors,
            hovertemplate=
            "<b>Indicator Variable: %{x}</b><br>Computed IV Value: %{y:.4f}<extra></extra>",
        ))

    for threshold, color, label in [
        (0.3, COMPLIANT_GREEN, "Strong Bound (0.30)"),
        (0.1, WARNING_AMBER, "Medium Limit (0.10)"),
        (0.5, CRITICAL_RED, "Leakage Cutoff (0.50)")
    ]:
        fig_iv.add_hline(y=threshold,
                         line_dash="dash",
                         line_color=color,
                         line_width=1,
                         annotation_text=label,
                         annotation_position="top left",
                         annotation_font_size=9,
                         annotation_font_color=color)

    fig_iv.update_layout(
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(tickfont=dict(size=9, color=FNB_DARK_TEXT),
                   gridcolor=BORDER_GRAY),
        yaxis=dict(title="Information Value Scale",
                   tickfont=dict(size=9, color=FNB_DARK_TEXT),
                   gridcolor=BORDER_GRAY,
                   zeroline=False),
    )
    apply_corporate_layout(fig_iv)
    st.plotly_chart(fig_iv, width='stretch')

    # ══════════════════════════════════════════════════════════
    # 3. INTERACTIVE RISK BINNING INSPECTION WORKBENCH
    # ══════════════════════════════════════════════════════════
    _sec("Interactive Binning Inspector Workbench", "WoE INSPECTOR")

    selected_var = st.selectbox(
        "Select Target Portfolio Indicator to Evaluate", features)

    bin_table, total_iv, _ = auto_bin_numeric(df, selected_var, target_col)
    stats = _feature_stats(df, selected_var, target_col)

    if bin_table.empty:
        st.warning(
            "Selected column contains insufficient variance distributions to successfully map out bin partitions."
        )
        return

    strength = _iv_strength(total_iv)

    st.markdown(
        f'<div class="feat-meta">'
        f'<span class="feat-name">{selected_var}</span>'
        f'{_badge(strength)}'
        f'<span style="font-size:0.8rem; color:{TEXT_MUTED};">'
        f'Distribution Mean: {stats["mean"]:.2f} &nbsp;|&nbsp; Missing Metric: {stats["missing_pct"]:.2f}% &nbsp;|&nbsp; Skew Parameter: {stats["skew"]:.2f}'
        f'</span>'
        f'<span class="feat-iv">Information Value Score = <strong>{total_iv:.4f}</strong></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if total_iv > 0.5:
        st.markdown(f"""
        <div class="leakage-alert">
            <div class="al-title">Potential Feature Leakage Risk Flagged</div>
            <div class="al-body">
                <strong>{selected_var}</strong> holds an intense Information Value score of <strong>{total_iv:.4f}</strong>. This parameter crosses the standard credit model leakage limit (0.50) and may represent an unapproved target proxy metric.
            </div>
        </div>
        """,
                    unsafe_allow_html=True)

    fmt_table = bin_table.rename(
        columns={
            "bin_label": "Bin Range Boundary Matrix",
            "count": "Record Frequency",
            "count_%": "Volume Share %",
            "goods": "Goods Logged",
            "bads": "Bads Logged",
            "bad_rate_%": "Observed Bad Rate %",
            "woe": "Weight of Evidence (WoE)",
            "iv": "IV Slice Contribution",
        })[[
            "Bin Range Boundary Matrix", "Record Frequency", "Volume Share %",
            "Goods Logged", "Bads Logged", "Observed Bad Rate %",
            "Weight of Evidence (WoE)", "IV Slice Contribution"
        ]]

    woe_styled = (fmt_table.style.format({
        "Volume Share %": "{:.2f}%",
        "Observed Bad Rate %": "{:.2f}%",
        "Weight of Evidence (WoE)": "{:.4f}",
        "IV Slice Contribution": "{:.4f}"
    }).background_gradient(
        subset=["Weight of Evidence (WoE)"],
        cmap="RdYlGn",
        vmin=-1.5,
        vmax=1.5).background_gradient(
            subset=["Observed Bad Rate %"], cmap="Reds", vmin=0,
            vmax=100).set_properties(**{
                "font-size": "0.82rem",
                "padding": "0.45rem 0.7rem"
            }).set_table_styles([{
                "selector":
                "th",
                "props": [("background-color", FNB_NAVY), ("color", WHITE),
                          ("font-size", "0.72rem"),
                          ("text-transform", "uppercase"),
                          ("letter-spacing", "0.05em")]
            }]).hide(axis="index"))

    st.dataframe(woe_styled, width='stretch')

    # ══════════════════════════════════════════════════════════
    # 4. FNB COMPLIANT VISUAL TREND PROFILE ANALYSIS
    # ══════════════════════════════════════════════════════════
    _sec("WoE Risk Trajectory and Empirical Default Distribution Profiles",
         "MONOTONICITY VALIDATION")

    col1, col2 = st.columns(2)

    with col1:
        fig_woe = go.Figure()
        fig_woe.add_trace(
            go.Bar(
                x=bin_table["bin_label"],
                y=bin_table["woe"],
                marker_color=[
                    FNB_TURQUOISE if w >= 0 else CRITICAL_RED
                    for w in bin_table["woe"]
                ],
                hovertemplate=
                "<b>Range: %{x}</b><br>WoE Score: %{y:.4f}<extra></extra>",
            ))
        fig_woe.add_trace(
            go.Scatter(x=bin_table["bin_label"],
                       y=bin_table["woe"],
                       mode="lines+markers",
                       line=dict(color=FNB_GOLD, width=2),
                       marker=dict(color=FNB_GOLD, size=6),
                       hoverinfo="skip"))
        fig_woe.add_hline(y=0, line_color=BORDER_GRAY, line_width=1)
        fig_woe.update_layout(
            plot_bgcolor=WHITE,
            paper_bgcolor=WHITE,
            height=280,
            showlegend=False,
            margin=dict(l=0, r=0, t=20, b=0),
            title=dict(text="Weight of Evidence (WoE) Across Risk Groups",
                       font=dict(size=11, color=FNB_NAVY),
                       x=0),
            xaxis=dict(tickfont=dict(size=9, color=FNB_DARK_TEXT),
                       gridcolor=BORDER_GRAY),
            yaxis=dict(title="WoE Log Scale",
                       tickfont=dict(size=9, color=FNB_DARK_TEXT),
                       gridcolor=BORDER_GRAY),
        )
        apply_corporate_layout(fig_woe)
        st.plotly_chart(fig_woe, width='stretch')

    with col2:
        fig_bad = go.Figure(
            go.Bar(
                x=bin_table["bin_label"],
                y=bin_table["bad_rate_%"],
                marker_color=FNB_NAVY,
                hovertemplate=
                "<b>Range: %{x}</b><br>Empirical Bad Rate: %{y:.2f}%<extra></extra>",
            ))
        fig_bad.add_hline(
            y=stats["overall_default_rate"],
            line_dash="dot",
            line_color=FNB_GOLD,
            line_width=1.5,
            annotation_text=
            f"Portfolio Benchmark Core Avg ({stats['overall_default_rate']:.2f}%)",
            annotation_position="top right",
            annotation_font_size=9,
            annotation_font_color=FNB_DARK_TEXT)
        fig_bad.update_layout(
            plot_bgcolor=WHITE,
            paper_bgcolor=WHITE,
            height=280,
            margin=dict(l=0, r=0, t=20, b=0),
            title=dict(text="Empirical Bad Rate (%) Volatility Tracking",
                       font=dict(size=11, color=FNB_NAVY),
                       x=0),
            xaxis=dict(tickfont=dict(size=9, color=FNB_DARK_TEXT),
                       gridcolor=BORDER_GRAY),
            yaxis=dict(title="Default Scale Percentage",
                       tickfont=dict(size=9, color=FNB_DARK_TEXT),
                       gridcolor=BORDER_GRAY),
        )
        apply_corporate_layout(fig_bad)
        st.plotly_chart(fig_bad, width='stretch')

    # ══════════════════════════════════════════════════════════
    # 5. FIXED LIVE COGNITIVE ANALYST DISCOVERY CORE PANELS
    # ══════════════════════════════════════════════════════════
    _sec("Automated Financial Credit Risk Commentary",
         "GEN-AI COGNITIVE ANALYST")

    with st.spinner(
            "Generating institutional credit summary text portfolio reviews..."
    ):
        ai_commentary = _generate_ai_insight(selected_var, bin_table, total_iv,
                                             strength, stats)

    st.markdown(f"""
    <div class="ai-panel">
        <div class="ai-hdr">Senior Portfolio Credit Risk Analyst Briefing</div>
        <div class="ai-body">{ai_commentary}</div>
    </div>
    """,
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.9rem; font-weight:700; color:#0A2540; margin-bottom:0.5rem;">Data-Driven Actionable Pipeline Inferences</p>',
        unsafe_allow_html=True)

    recs_list = _recommendations(selected_var, total_iv, strength, bin_table,
                                 stats)

    if recs_list:
        for r in recs_list:
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-type">{r['type']}</div>
                <div class="rec-title">{r['title']}</div>
                <div class="rec-body">{r['body']}</div>
            </div>
            """,
                        unsafe_allow_html=True)
    else:
        st.info(
            "No alignment anomalies discovered inside current metric row distribution."
        )
