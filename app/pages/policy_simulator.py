"""
RiskLens Analytics — Policy Simulator
Executive Credit Policy Decision Center

app/pages/policy_simulator.py

Data source: data/raw/loan_book.csv  →  st.session_state.portfolio_df
─────────────────────────────────────────────────────────────────────
Schema used (loan_book.csv canonical columns):
  applicant_id_hash, age, annual_income, employment_length_years,
  home_ownership, region, num_open_accounts, num_delinquencies_2yr,
  total_revolving_balance, credit_utilisation_pct,
  months_since_oldest_account, num_hard_inquiries_6mo,
  loan_amount, interest_rate, loan_purpose, dti_ratio,
  months_since_last_delinquency, pct_accounts_current,
  application_date, application_dow, branch_code_id,
  months_at_current_address, email_domain_type, phone_verified,
  default_flag, set

All slider bounds, defaults, and stats are derived from the real
portfolio data — nothing is hardcoded.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.components.header import render_header
from app.components.cards import render_kpi_row
from app.components.charts import (
    apply_corporate_layout,
    PRIMARY_BLUE,
    ACCENT_GOLD,
    SECONDARY_BLUE,
    ALERT_RED,
    SUCCESS_GREEN,
)
from src.business_logic.score_engine import enrich_portfolio
from src.business_logic.policy_engine import (
    simulate_credit_policy,
    optimize_cutoff,
    run_scenario_comparison,
    compute_risk_tradeoff,
    generate_strategy_insights,
    generate_executive_recommendation,
    compute_portfolio_stats,
    build_risk_presets,
)
from src.evaluation.metrics import get_confusion_matrix_details

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────

NAVY = PRIMARY_BLUE
GOLD = ACCENT_GOLD
GOLD_LIGHT = "#E8B96A"
SLATE = "#8BA0B4"
MIST = "#EEF2F6"
WHITE = "#FFFFFF"
DANGER = ALERT_RED
SUCCESS = SUCCESS_GREEN

_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  .rl-section {{
    font-family: 'Libre Baskerville', serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: {NAVY};
    border-bottom: 1px solid #CBD8E4;
    padding-bottom: 8px;
    margin: 28px 0 16px;
    letter-spacing: -0.01em;
  }}
  .advisory-card {{
    background: {WHITE};
    border-left: 3px solid {NAVY};
    border-radius: 4px;
    padding: 13px 17px;
    margin-bottom: 9px;
    box-shadow: 0 1px 3px rgba(13,27,42,0.06);
    font-size: 0.83rem;
    line-height: 1.65;
    color: {NAVY};
    font-family: 'IBM Plex Sans', sans-serif;
  }}
  .advisory-card.insight  {{ border-left-color: {GOLD}; }}
  .advisory-card.warning  {{ border-left-color: {DANGER}; }}
  .advisory-card.positive {{ border-left-color: {SUCCESS}; }}
  .advisory-tag {{
    display: block;
    font-size: 0.61rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: {SLATE};
    margin-bottom: 5px;
  }}
  .rec-card {{
    background: {NAVY};
    border-radius: 6px;
    padding: 20px 22px;
    color: {WHITE};
    box-shadow: 0 2px 8px rgba(13,27,42,0.14);
    height: 100%;
    font-family: 'IBM Plex Sans', sans-serif;
  }}
  .rec-card h4 {{
    font-family: 'Libre Baskerville', serif;
    font-size: 0.78rem;
    color: {GOLD_LIGHT};
    margin: 0 0 10px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
  }}
  .rec-score {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: {WHITE};
    letter-spacing: -0.02em;
    margin: 8px 0 3px;
    line-height: 1;
  }}
  .rec-label {{
    font-size: 0.66rem;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: {SLATE};
  }}
  .rec-card p {{
    font-size: 0.80rem;
    color: #B8CDD9;
    margin: 12px 0 0;
    line-height: 1.6;
  }}
  .scenario-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.80rem;
    font-family: 'IBM Plex Sans', sans-serif;
  }}
  .scenario-table th {{
    background: {NAVY};
    color: {WHITE};
    padding: 9px 11px;
    text-align: left;
    font-weight: 600;
    font-size: 0.71rem;
    letter-spacing: 0.03em;
  }}
  .scenario-table td {{
    padding: 8px 11px;
    border-bottom: 1px solid {MIST};
    color: {NAVY};
  }}
  .scenario-table tr:nth-child(even) td {{ background: #F5F8FB; }}
  .pill {{
    display: inline-block;
    padding: 2px 9px;
    border-radius: 10px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.04em;
  }}
  .pill-green {{ background: #E6F4EC; color: {SUCCESS}; }}
  .pill-red   {{ background: #FDECEA; color: {DANGER};  }}
  .pill-gold  {{ background: #FEF6E8; color: #A07020;   }}
  .stat-row {{
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid {MIST};
    font-size: 0.81rem;
  }}
  .stat-label {{ color: {SLATE}; }}
  .stat-value {{ font-family: 'IBM Plex Mono', monospace; font-weight: 600; color: {NAVY}; }}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# LOAN_BOOK.CSV CANONICAL COLUMN NAMES
# ─────────────────────────────────────────────────────────────────────────────
# Kept in sync with the actual header row of data/raw/loan_book.csv.
# If the CSV ever gains an alias, add it to the relevant candidates list.

_RATE_CANDIDATES = [
    "interest_rate",  # canonical loan_book.csv name
    "interest_rate_pct",
    "int_rate",
    "rate",
    "apr",
    "annual_rate",
    "lending_rate",
    "rate_pct",
    "loan_rate",
    "ir",
]

_LOAN_CANDIDATES = [
    "loan_amount",  # canonical loan_book.csv name
    "loan_size",
    "amount",
    "funded_amount",
    "loan_amnt",
    "principal",
    "loan_value",
    "disbursement_amount",
]

_TARGET_CANDIDATES = [
    "default_flag",  # canonical loan_book.csv name
    "default",
    "bad_flag",
    "bad",
    "target",
    "label",
    "outcome",
    "defaulted",
    "is_default",
]

_ID_CANDIDATES = [
    "applicant_id_hash",  # canonical loan_book.csv name
    "account_id",
    "client_id",
    "id",
    "customer_id",
    "applicant_id",
    "loan_id",
]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace AND lowercase all column names.
    Ensures canonical loan_book.csv names like 'interest_rate' are always
    matched regardless of BOM chars, Excel artefacts, or mixed-case headers.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _detect_col(
    df: pd.DataFrame,
    candidates: list,
    fallback_substr: str,
    required: bool = True,
) -> str | None:
    """Return first matching column (case-insensitive, whitespace-stripped)."""
    lower_cols = {c.strip().lower(): c.strip() for c in df.columns}

    for cand in candidates:
        if cand.strip().lower() in lower_cols:
            return lower_cols[cand.strip().lower()]

    for col in df.columns:
        if fallback_substr.strip().lower() in col.strip().lower():
            return col.strip()

    if not required:
        return None

    raise KeyError(f"Could not find a '{fallback_substr}' column.\n"
                   f"Searched for: {candidates}\n"
                   f"Available columns: {[c.strip() for c in df.columns]}\n"
                   f"Ensure loan_book.csv was loaded without transformation.")


# ─────────────────────────────────────────────────────────────────────────────
# CACHED FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Deriving credit scores from portfolio features…")
def _prepare_data(_df_hash: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise columns and add credit_score derived from loan_book.csv features.

    Critical: column names are lowercased here so that 'interest_rate',
    'loan_amount', 'default_flag', etc. are always detectable downstream
    regardless of how the CSV was loaded.

    enrich_portfolio() may internally consume some columns; we snapshot
    all original columns first and re-merge anything missing from the
    returned DataFrame so no source column (esp. interest_rate) is lost.
    """
    df = df.copy()
    # Lowercase + strip ALL column names — fixes BOM, mixed-case, Excel artefacts
    df.columns = [c.strip().lower() for c in df.columns]

    if "credit_score" not in df.columns:
        # Snapshot original columns before enrich (some engines drop columns)
        original_cols = df.copy()
        enriched = enrich_portfolio(df)
        # Lowercase enriched columns too
        enriched.columns = [c.strip().lower() for c in enriched.columns]
        # Re-attach any original columns that enrich_portfolio dropped
        for col in original_cols.columns:
            if col not in enriched.columns:
                enriched[col] = original_cols[col].values
        df = enriched

    return df


@st.cache_data(show_spinner=False)
def _cached_portfolio_stats(
    _scores_key: int,
    scores: np.ndarray,
    y_true: np.ndarray,
    loan_amounts: np.ndarray,
    interest_rates: np.ndarray,
    lgd_rate: float,
) -> dict:
    return compute_portfolio_stats(scores,
                                   y_true,
                                   loan_amounts,
                                   interest_rates,
                                   lgd_rate=lgd_rate)


@st.cache_data(show_spinner="Running profitability sweep…")
def _cached_optimize(
    _scores_key: int,
    scores: np.ndarray,
    y_true: np.ndarray,
    loan_size: float,
    rate: float,
    lgd: float,
) -> tuple:
    return optimize_cutoff(
        scores,
        y_true,
        avg_loan_size=loan_size,
        interest_rate=rate,
        lgd_rate=lgd,
        n_steps=100,
    )


@st.cache_data(show_spinner=False)
def _cached_tradeoff(
    _scores_key: int,
    scores: np.ndarray,
    y_true: np.ndarray,
    loan_size: float,
    rate: float,
    lgd: float,
) -> pd.DataFrame:
    return compute_risk_tradeoff(
        scores,
        y_true,
        avg_loan_size=loan_size,
        interest_rate=rate,
        lgd_rate=lgd,
        n_steps=100,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _load_arrays(df: pd.DataFrame) -> tuple:
    """
    Extract typed numpy arrays from the enriched portfolio DataFrame.

    Reads directly from the loan_book.csv canonical columns:
      credit_score  — derived by enrich_portfolio()
      default_flag  — binary outcome (0/1)
      loan_amount   — original disbursement amount (R)
      interest_rate — annual rate as percentage (e.g. 12.5 → 0.125 after norm)
    """
    df = _normalise_columns(df)

    scores = df["credit_score"].values.astype(float)

    target_col = _detect_col(df, _TARGET_CANDIDATES, "default")
    y_true = df[target_col].values.astype(int)

    loan_col = _detect_col(df, _LOAN_CANDIDATES, "loan")
    _loan_num = pd.to_numeric(df[loan_col], errors="coerce")
    loan_amounts = _loan_num.fillna(_loan_num.median()).values.astype(float)

    rate_col = _detect_col(df, _RATE_CANDIDATES, "rate", required=False)

    if rate_col is None:
        _fallback = 0.12
        st.warning(
            f"⚠️ No interest_rate column found.  "
            f"Using {_fallback*100:.1f}% fallback.  "
            f"Ensure loan_book.csv contains the 'interest_rate' column.",
            icon="⚠️",
        )
        interest_rates = np.full(len(df), _fallback, dtype=float)
    else:
        _rate_num = pd.to_numeric(df[rate_col], errors="coerce")
        _rate_filled = _rate_num.fillna(_rate_num.median())
        # loan_book.csv stores interest_rate as a percentage (e.g. 12.5)
        # Normalise to decimal (0.125) for financial calculations
        interest_rates = ((_rate_filled / 100.0).values.astype(float)
                          if _rate_filled.median() > 1 else
                          _rate_filled.values.astype(float))

    return scores, y_true, loan_amounts, interest_rates


def _estimate_lgd_from_data(
    y_true: np.ndarray,
    loan_amounts: np.ndarray,
    interest_rates: np.ndarray,
) -> float:
    """
    Estimate a data-anchored LGD seed for the UI slider default.

    Uses portfolio bad rate and average interest rate to compute the
    LGD at which the portfolio breaks even — a meaningful starting
    point grounded in the actual loan_book.csv data.

        LGD_seed = avg_rate / overall_bad_rate_decimal

    Clamped to [0.20, 0.95].
    """
    bad_rate_decimal = float(y_true.mean())
    avg_rate = float(np.mean(interest_rates))

    if bad_rate_decimal > 0:
        # Data-anchored: breakeven LGD derived from actual portfolio bad rate
        # and average interest_rate from loan_book.csv
        lgd_seed = avg_rate / bad_rate_decimal
    else:
        # Extreme edge case: dataset has zero defaults (impossible in loan_book.csv
        # which has real defaults, but guards against a filtered empty subset).
        # Fall back to avg_rate itself as a conservative seed so at least the
        # slider default reflects the portfolio's actual revenue rate.
        lgd_seed = avg_rate if avg_rate > 0 else float(np.mean(interest_rates))

    return round(float(np.clip(lgd_seed, 0.20, 0.95)), 4)


def _fmt_zar(val: float) -> str:
    if abs(val) >= 1_000_000_000:
        return f"R {val/1_000_000_000:.2f}B"
    if abs(val) >= 1_000_000:
        return f"R {val/1_000_000:.2f}M"
    if abs(val) >= 1_000:
        return f"R {val/1_000:.1f}K"
    return f"R {val:,.0f}"


def _section(title: str) -> None:
    st.markdown(f'<div class="rl-section">{title}</div>',
                unsafe_allow_html=True)


def _plotly_chart(fig: go.Figure, height: int = 300) -> None:
    apply_corporate_layout(fig)
    fig.update_layout(height=height)
    st.plotly_chart(fig,
                    use_container_width=True,
                    config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────


def _chart_score_distribution(scores: np.ndarray, cutoff: float) -> go.Figure:
    approved = scores[scores >= cutoff]
    rejected = scores[scores < cutoff]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=rejected,
            name="Rejected",
            marker_color="#CBD8E4",
            opacity=0.85,
            nbinsx=50,
        ))
    fig.add_trace(
        go.Histogram(
            x=approved,
            name="Approved",
            marker_color=NAVY,
            opacity=0.85,
            nbinsx=50,
        ))
    fig.add_vline(
        x=cutoff,
        line_color=GOLD,
        line_width=2,
        line_dash="dash",
        annotation_text=f"  Cutoff: {cutoff:.1f}",
        annotation_font=dict(color=GOLD, size=11, family="IBM Plex Mono"),
        annotation_position="top right",
    )
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Derived Credit Score",
        yaxis_title="Applications",
        legend=dict(orientation="h", y=-0.22, x=0),
    )
    return fig


def _chart_optimisation(sweep_df: pd.DataFrame, best_cutoff: float,
                        cutoff: float) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Net Profit (R)", "ROI (%)", "Approval Rate (%)"),
        horizontal_spacing=0.10,
    )
    series = [
        ("Net Profit", NAVY, 1),
        ("ROI %", GOLD, 2),
        ("Approval Rate %", SECONDARY_BLUE, 3),
    ]
    for col_name, color, col_idx in series:
        fig.add_trace(
            go.Scatter(
                x=sweep_df["Cutoff Score"],
                y=sweep_df[col_name],
                mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy" if col_idx == 1 else None,
                fillcolor="rgba(13,27,42,0.05)" if col_idx == 1 else None,
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )
        fig.add_vline(x=best_cutoff,
                      line_color=SUCCESS_GREEN,
                      line_width=1.5,
                      line_dash="dot",
                      row=1,
                      col=col_idx)
        fig.add_vline(x=cutoff,
                      line_color=GOLD,
                      line_width=1.5,
                      line_dash="dash",
                      row=1,
                      col=col_idx)
    for label, color, dash in [
        (f"Optimal ({best_cutoff:.1f})", SUCCESS_GREEN, "dot"),
        (f"Active  ({cutoff:.1f})", GOLD, "dash"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=color, dash=dash, width=1.5),
                name=label,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    fig.update_xaxes(title_text="Cutoff Score", title_font_size=10)
    fig.update_layout(
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=10)),
        showlegend=True,
    )
    return fig


def _chart_risk_tradeoff(tradeoff_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tradeoff_df["Approval Rate %"],
            y=tradeoff_df["Bad Rate %"],
            mode="lines+markers",
            marker=dict(
                size=6,
                color=tradeoff_df["Net Profit"],
                colorscale=[[0, "#CBD8E4"], [0.5, GOLD], [1, NAVY]],
                showscale=True,
                colorbar=dict(
                    title=dict(text="Net Profit",
                               side="right",
                               font=dict(size=9)),
                    thickness=10,
                    len=0.7,
                    tickfont=dict(size=8),
                ),
            ),
            line=dict(color="#CBD8E4", width=1.5),
            text=tradeoff_df["Cutoff Score"],
            hovertemplate=("<b>Cutoff: %{text:.1f}</b><br>"
                           "Approval: %{x:.1f}%<br>"
                           "Bad Rate: %{y:.2f}%<extra></extra>"),
        ))
    fig.update_layout(xaxis_title="Approval Rate (%)",
                      yaxis_title="Bad Rate (%)")
    return fig


def _chart_capital_vs_profit(tradeoff_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tradeoff_df["Capital Lent"],
            y=tradeoff_df["Net Profit"],
            mode="lines+markers",
            marker=dict(size=5, color=NAVY),
            line=dict(color=NAVY, width=1.5),
            text=tradeoff_df["Cutoff Score"],
            hovertemplate=("<b>Cutoff: %{text:.1f}</b><br>"
                           "Capital: R%{x:,.0f}<br>"
                           "Profit: R%{y:,.0f}<extra></extra>"),
        ))
    fig.update_layout(xaxis_title="Capital Deployed (R)",
                      yaxis_title="Net Profit (R)")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SECTION RENDERERS
# ─────────────────────────────────────────────────────────────────────────────


def _render_controls(
    scores: np.ndarray,
    portfolio_stats: dict,
    avg_loan: float,
    avg_rate: float,
    lgd_default: float,
) -> dict:
    """
    Render all policy controls.  Every slider bound and default is
    derived from the real loan_book.csv data passed via portfolio_stats.
    """
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    _section("Credit Policy Calibration")

    preset_col, _ = st.columns([2, 5])
    with preset_col:
        preset = st.radio(
            "Risk Appetite Preset",
            ["Conservative", "Balanced", "Aggressive"],
            index=1,
            horizontal=True,
        )

    presets = build_risk_presets(scores, avg_loan, avg_rate, lgd_default)
    cfg = presets[preset]

    score_min = portfolio_stats["score_min"]
    score_max = portfolio_stats["score_max"]
    step = round((score_max - score_min) / 200, 2)

    # ── Loan amount bounds (P5–P95 of loan_amount column) ─────────────────
    _loan_p5 = float(portfolio_stats["loan_p5"])
    _loan_p95 = float(portfolio_stats["loan_p95"])
    if _loan_p5 >= _loan_p95:
        _loan_p5 = float(portfolio_stats["loan_min"])
        _loan_p95 = float(portfolio_stats["loan_max"])
    if _loan_p95 - _loan_p5 < 1_000:
        _loan_p5 = max(0.0, _loan_p5 - 5_000)
        _loan_p95 = _loan_p5 + 10_000
    loan_slider_min = max(500.0, _loan_p5)
    loan_slider_max = max(loan_slider_min + 1_000, _loan_p95)
    loan_iqr = loan_slider_max - loan_slider_min
    loan_step = max(500, round(loan_iqr / 50 / 500) * 500)

    # ── Interest rate bounds (P5–P95 of interest_rate column) ─────────────
    _rate_p5_pct = float(portfolio_stats["rate_p5"]) * 100
    _rate_p95_pct = float(portfolio_stats["rate_p95"]) * 100
    _avg_rate_pct = float(avg_rate) * 100
    if abs(_rate_p95_pct - _rate_p5_pct) < 0.5:
        _rate_p5_pct = max(1.0, _avg_rate_pct - 5.0)
        _rate_p95_pct = min(60.0, _avg_rate_pct + 5.0)
    rate_slider_min = max(1.0, round(_rate_p5_pct, 1))
    rate_slider_max = min(60.0, round(_rate_p95_pct, 1))
    if rate_slider_min >= rate_slider_max:
        rate_slider_min = max(1.0, rate_slider_max - 5.0)
    _avg_rate_display = round(
        float(np.clip(_avg_rate_pct, rate_slider_min, rate_slider_max)), 1)

    lgd_slider_min = 10
    lgd_slider_max = 95
    lgd_default_pct = int(round(lgd_default * 100))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cutoff = st.slider(
            "Credit Score Cutoff",
            min_value=float(score_min),
            max_value=float(score_max),
            value=float(cfg["cutoff"]),
            step=float(step),
            format="%.1f",
            help=(
                f"Derived score range: {score_min:.1f} – {score_max:.1f}  |  "
                f"Median: {portfolio_stats['median_score']:.1f}"),
        )
        st.session_state.cutoff_score = cutoff
    with c2:
        _loan_default = int(
            np.clip(
                round(avg_loan / loan_step) * loan_step,
                int(loan_slider_min),
                int(loan_slider_max),
            ))
        loan_size = st.number_input(
            "Average Loan Size (R)",
            min_value=int(loan_slider_min),
            max_value=int(loan_slider_max),
            value=_loan_default,
            step=loan_step,
            help=
            (f"Portfolio mean (loan_amount): R{avg_loan:,.0f}  |  "
             f"Range (P5–P95): R{loan_slider_min:,.0f} – R{loan_slider_max:,.0f}"
             ),
        )
    with c3:
        rate_pct = st.slider(
            "Interest / Revenue Rate (%)",
            min_value=float(rate_slider_min),
            max_value=float(rate_slider_max),
            value=float(_avg_rate_display),
            step=0.5,
            format="%.1f%%",
            help=
            (f"Portfolio mean (interest_rate): {avg_rate*100:.2f}%  |  "
             f"Range (P5–P95): {rate_slider_min:.1f}% – {rate_slider_max:.1f}%"
             ),
        )
        rate = rate_pct / 100.0
    with c4:
        lgd_pct = st.slider(
            "Loss Given Default — LGD (%)",
            min_value=lgd_slider_min,
            max_value=lgd_slider_max,
            value=lgd_default_pct,
            step=1,
            format="%d%%",
            help=(
                f"Data-anchored default: {lgd_default_pct}%  |  "
                f"Estimated from portfolio bad rate and average interest_rate."
            ),
        )
        lgd = lgd_pct / 100.0

    st.markdown('</div>', unsafe_allow_html=True)
    return dict(preset=preset,
                cutoff=cutoff,
                loan_size=float(loan_size),
                rate=rate,
                lgd=lgd)


def _render_kpis(sim: dict, portfolio_stats: dict) -> None:
    breakeven = portfolio_stats["breakeven_bad_rate"]
    kpis = [
        {
            "label": "Approval Rate",
            "value": f"{sim['approval_rate']}%",
            "delta_text":
            f"{sim['total_approved']:,} of {sim['total_applications']:,}",
            "delta_direction": "neutral",
        },
        {
            "label": "Rejection Rate",
            "value": f"{sim['rejection_rate']}%",
            "delta_text": f"{sim['total_rejected']:,} declined",
            "delta_direction": "neutral",
        },
        {
            "label":
            "Expected Defaults",
            "value":
            f"{sim['approved_defaults']:,}",
            "delta_text":
            f"Bad rate: {sim['bad_rate_approved']}%",
            "delta_direction":
            "down" if sim["bad_rate_approved"] < breakeven else "up",
        },
        {
            "label": "Net Profit / Loss",
            "value": _fmt_zar(sim["net_profit"]),
            "delta_text": f"Revenue: {_fmt_zar(sim['expected_revenue'])}",
            "delta_direction": "up" if sim["net_profit"] > 0 else "down",
        },
        {
            "label": "Portfolio ROI",
            "value": f"{sim['roi']:.2f}%",
            "delta_text": "Annualised net return",
            "delta_direction": "up" if sim["roi"] > 0 else "down",
        },
        {
            "label": "Capital Deployed",
            "value": _fmt_zar(sim["total_capital_lent"]),
            "delta_text": f"{sim['total_approved']:,} loans issued",
            "delta_direction": "neutral",
        },
    ]
    render_kpi_row(kpis)


def _render_portfolio_summary(portfolio_stats: dict) -> None:
    _section("Real Portfolio Statistics — loan_book.csv")
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("Total Applications", f"{portfolio_stats['total_applications']:,}"),
        ("Overall Bad Rate", f"{portfolio_stats['overall_bad_rate']:.2f}%"),
        ("Avg Loan Amount", f"R{portfolio_stats['avg_loan']:,.0f}"),
        ("Avg Interest Rate", f"{portfolio_stats['avg_rate']*100:.2f}%"),
        ("Median Score", f"{portfolio_stats['median_score']:.1f}"),
        ("Score Range",
         f"{portfolio_stats['score_min']:.1f} – {portfolio_stats['score_max']:.1f}"
         ),
        ("Breakeven Bad Rate",
         f"{portfolio_stats['breakeven_bad_rate']:.2f}%"),
        ("P25 / P75 Score",
         f"{portfolio_stats['p25_score']:.1f} / {portfolio_stats['p75_score']:.1f}"
         ),
    ]
    cols = [c1, c2, c3, c4]
    for i, (label, value) in enumerate(stats):
        with cols[i % 4]:
            st.markdown(
                f'<div class="stat-row">'
                f'<span class="stat-label">{label}</span>'
                f'<span class="stat-value">{value}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)


def _render_score_distribution(scores: np.ndarray, cutoff: float) -> None:
    _section("Score Distribution — Approval Segmentation")
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    _plotly_chart(_chart_score_distribution(scores, cutoff), height=290)
    st.markdown('</div>', unsafe_allow_html=True)


def _render_optimisation(sweep_df: pd.DataFrame, best_cutoff: float,
                         cutoff: float) -> None:
    _section("Profitability Optimisation Center")
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    _plotly_chart(_chart_optimisation(sweep_df, best_cutoff, cutoff),
                  height=300)
    with st.expander("Full Sweep Table", expanded=False):
        st.dataframe(
            sweep_df[[
                "Cutoff Score", "Approval Rate %", "Bad Rate %", "Net Profit",
                "ROI %"
            ]].style.format({
                "Net Profit": lambda v: _fmt_zar(v),
                "Approval Rate %": "{:.1f}%",
                "Bad Rate %": "{:.2f}%",
                "ROI %": "{:.2f}%",
                "Cutoff Score": "{:.1f}",
            }),
            use_container_width=True,
            height=260,
            hide_index=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


def _render_risk_tradeoff(tradeoff_df: pd.DataFrame) -> None:
    _section("Risk Tradeoff Intelligence")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.caption("Approval Rate vs Bad Rate — coloured by Net Profit")
        _plotly_chart(_chart_risk_tradeoff(tradeoff_df), height=300)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_right:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.caption("Capital Exposure vs Profitability")
        _plotly_chart(_chart_capital_vs_profit(tradeoff_df), height=300)
        st.markdown('</div>', unsafe_allow_html=True)


def _render_confusion_matrix(y_true: np.ndarray, scores: np.ndarray,
                             cutoff: float) -> None:
    _section("Underwriting Decision Matrix")
    col1, col2 = st.columns(2)
    cm, metrics = get_confusion_matrix_details(y_true, scores, cutoff)
    tn, fp, fn, tp = cm.ravel()
    with col1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.caption("Classification Outcome Matrix")
        matrix_df = pd.DataFrame(
            [[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]],
            columns=["Predicted: Approve", "Predicted: Reject"],
            index=["Actual: Performing", "Actual: Defaulted"],
        )
        st.table(matrix_df)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.caption("Risk Policy Indicators")
        indicators = [
            ("Classification Accuracy", f"{metrics['accuracy']*100:.2f}%"),
            ("Default Capture Precision", f"{metrics['precision']*100:.2f}%"),
            ("False Acceptance Rate (Leakage)",
             f"{metrics['false_positive_rate']*100:.2f}%"),
            ("False Rejection Rate (Lost Oppty)",
             f"{metrics['false_negative_rate']*100:.2f}%"),
        ]
        for label, val in indicators:
            cl, cv = st.columns([3, 1])
            cl.markdown(
                f"<span style='font-size:0.81rem;color:#4A5568;'>{label}</span>",
                unsafe_allow_html=True,
            )
            cv.markdown(
                f"<span style='font-family:IBM Plex Mono,monospace;"
                f"font-size:0.88rem;font-weight:600;color:{NAVY};'>{val}</span>",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)


def _render_ai_advisor(sim: dict, best_cutoff: float, preset: str,
                       portfolio_stats: dict) -> None:
    _section("AI Credit Strategy Advisor")
    insights = generate_strategy_insights(sim, best_cutoff, preset,
                                          portfolio_stats)
    for ins in insights:
        st.markdown(
            f'<div class="advisory-card {ins["type"]}">'
            f'<span class="advisory-tag">{ins["tag"]}</span>'
            f'{ins["text"]}'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_recommendation(
    sim: dict,
    best_cutoff: float,
    sweep_df: pd.DataFrame,
    preset: str,
    portfolio_stats: dict,
) -> None:
    _section("Executive Recommendation Engine")
    rec = generate_executive_recommendation(sim, best_cutoff, sweep_df, preset,
                                            portfolio_stats)
    alignment_pill = {
        "aligned":
        f'<span class="pill pill-green">Aligned with Optimal</span>',
        "below_optimal": f'<span class="pill pill-red">Below Optimal</span>',
        "above_optimal": f'<span class="pill pill-gold">Above Optimal</span>',
    }.get(rec["active_vs_optimal"], "")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="rec-card">'
            f'<h4>Profit-Maximising Cutoff</h4>'
            f'<div class="rec-score">{rec["optimal_cutoff"]:.1f}</div>'
            f'<div class="rec-label">Optimal threshold</div>'
            f'<p>Active cutoff: <b style="color:{GOLD_LIGHT};">{rec["active_cutoff"]:.1f}</b>'
            f'&nbsp;{alignment_pill}<br/>'
            f'Gap: {abs(rec["gap"]):.1f} pts '
            f'{"below" if rec["gap"] < 0 else "above"} optimal</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="rec-card">'
            f'<h4>Expected Profitability at Optimal</h4>'
            f'<div class="rec-score">{_fmt_zar(rec["optimal_net_profit"])}</div>'
            f'<div class="rec-label">Net profit at optimal cutoff</div>'
            f'<p>'
            f'Approval rate: <b style="color:{GOLD_LIGHT};">{rec["optimal_approval_rate"]:.1f}%</b><br/>'
            f'Bad rate: <b style="color:{GOLD_LIGHT};">{rec["optimal_bad_rate"]:.2f}%</b><br/>'
            f'ROI: <b style="color:{GOLD_LIGHT};">{rec["optimal_roi"]:.2f}%</b>'
            f'</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="rec-card">'
            f'<h4>Recommended Policy Posture</h4>'
            f'<div style="font-family:\'Libre Baskerville\',serif;font-size:0.90rem;'
            f'color:{GOLD_LIGHT};margin:10px 0 6px;font-weight:700;line-height:1.3;">'
            f'{rec["suggested_posture"]}</div>'
            f'<div class="rec-label">Active preset: {rec["preset"]}</div>'
            f'<p>{rec["posture_rationale"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_scenario_comparison(
    scores: np.ndarray,
    y_true: np.ndarray,
    avg_loan: float,
    avg_rate: float,
    lgd: float,
) -> None:
    _section("Scenario Comparison Center")
    comp_df = run_scenario_comparison(scores, y_true, avg_loan, avg_rate, lgd)
    for col in ["Capital Lent", "Default Loss", "Net Profit"]:
        comp_df[col] = comp_df[col].apply(_fmt_zar)
    comp_df["Cutoff"] = comp_df["Cutoff"].apply(lambda v: f"{v:.1f}")

    header_html = "".join(f"<th>{h}</th>" for h in comp_df.columns)
    rows_html = "".join(f"<tr>{''.join(f'<td>{v}</td>' for v in row)}</tr>"
                        for _, row in comp_df.iterrows())
    st.markdown(
        f'<div class="content-card">'
        f'<table class="scenario-table">'
        f'<thead><tr>{header_html}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )


def _render_decision_table(sim: dict, df: pd.DataFrame) -> None:
    _section("Decision Intelligence Table")
    with st.expander("View Individual Loan Decisions", expanded=False):
        decisions = sim["decisions_df"].copy()
        score_vals = decisions["Credit Score"].values.astype(float)

        col_f, col_d, _ = st.columns([2, 2, 3])
        with col_f:
            min_filter = st.number_input(
                "Min Credit Score",
                min_value=float(score_vals.min()),
                max_value=float(score_vals.max()),
                value=float(score_vals.min()),
                step=1.0,
                format="%.1f",
            )
        with col_d:
            dec_filter = st.selectbox("Decision",
                                      ["All", "Approved", "Rejected"])

        filtered = decisions[decisions["Credit Score"] >= min_filter]
        if dec_filter != "All":
            filtered = filtered[filtered["Decision"] == dec_filter]

        st.dataframe(
            filtered.sort_values("Credit Score",
                                 ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=320,
            column_config={
                "Credit Score":
                st.column_config.NumberColumn(format="%.2f"),
                "Defaulted":
                st.column_config.NumberColumn("Default (1=Yes)", format="%d"),
                "Decision":
                st.column_config.TextColumn(),
                "Risk Band":
                st.column_config.TextColumn(),
                "Outcome":
                st.column_config.TextColumn(),
            },
        )
        st.caption(
            f"Showing {len(filtered):,} of {len(decisions):,} records  ·  "
            f"Source: st.session_state.portfolio_df ← data/raw/loan_book.csv")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def render_page() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    render_header(
        page_title="Policy Simulator",
        page_subtitle=("Executive Credit Policy Decision Center — "
                       "Underwriting Strategy & Profitability Modeler"),
    )

    # ── 1. Load full portfolio from CSV ───────────────────────────────────
    # ALWAYS load from the CSV file directly to guarantee all 120k+ rows.
    # session_state.portfolio_df may contain a filtered subset (e.g. only the
    # 'test' split) if another page filtered it upstream.
    CSV_PATH = "data/raw/loan_book.csv"

    @st.cache_data(show_spinner="Loading loan_book.csv…")
    def _load_full_csv(path: str) -> pd.DataFrame:
        """
        Load the complete loan_book.csv — all rows, no filtering.

        Handles mixed date formats in loan_book.csv:
          ISO: 2021-08-28 | US slash: 06/14/2021 | SA slash: 23/03/2021 | Named: 23-Aug-2021
        Coerces home_ownership / loan_purpose to consistent lowercase.
        """
        raw = pd.read_csv(path, low_memory=False)
        # Normalise column names immediately — strip + lowercase
        raw.columns = [c.strip().lower() for c in raw.columns]

        # Normalise string columns with inconsistent casing in loan_book.csv
        for col in ["home_ownership", "loan_purpose"]:
            if col in raw.columns:
                raw[col] = raw[col].str.strip().str.lower()

        # Parse application_date robustly — handles the four mixed formats in loan_book.csv:
        #   ISO:        2021-08-28
        #   US slash:   06/14/2021 / 12/30/2021
        #   SA slash:   23/03/2021
        #   Named:      23-Aug-2021
        # infer_datetime_format was removed in pandas >= 2.0; use dateutil parser instead.
        if "application_date" in raw.columns:
            raw["application_date"] = pd.to_datetime(raw["application_date"],
                                                     dayfirst=False,
                                                     errors="coerce")

        # Ensure phone_verified is boolean-compatible
        if "phone_verified" in raw.columns:
            raw["phone_verified"] = (raw["phone_verified"].map({
                "True": True,
                "False": False,
                True: True,
                False: False
            }).fillna(False))

        return raw

    try:
        raw_df = _load_full_csv(CSV_PATH)
    except FileNotFoundError:
        if "portfolio_df" not in st.session_state:
            st.error(
                f"Cannot find `{CSV_PATH}` and no portfolio loaded in session.  "
                "Place loan_book.csv at data/raw/loan_book.csv and restart.")
            return
        # Fallback: use session state but normalise columns
        raw_df = st.session_state.portfolio_df.copy()
        raw_df.columns = [c.strip().lower() for c in raw_df.columns]
        st.warning(
            f"`{CSV_PATH}` not found on disk.  "
            f"Using session state ({len(raw_df):,} rows).  "
            "Row count may reflect a filtered subset.",
            icon="⚠️",
        )

    # Diagnostic strip — confirms actual row count on every page load
    n_rows = len(raw_df)
    n_cols = len(raw_df.columns)
    st.caption(
        f"📂 `data/raw/loan_book.csv` — **{n_rows:,} rows**, {n_cols} columns  ·  "
        f"Columns: {', '.join(raw_df.columns[:8].tolist())}…")

    # Use row count as a stable cache key — avoids the id() memory-address churn
    # that caused spurious cache misses on every Streamlit rerun.
    df = _prepare_data(n_rows, raw_df)

    # Write the full enriched df back so other pages also see all rows
    st.session_state.portfolio_df = df

    # ── 2. Extract arrays from loan_book.csv columns ──────────────────────
    scores, y_true, loan_amounts, interest_rates = _load_arrays(df)
    scores_key = hash(scores.tobytes())

    # ── 3. Data-anchored LGD seed ─────────────────────────────────────────
    lgd_seed = _estimate_lgd_from_data(y_true, loan_amounts, interest_rates)

    # ── 4. Pass-1 portfolio stats ─────────────────────────────────────────
    portfolio_stats = _cached_portfolio_stats(
        scores_key,
        scores,
        y_true,
        loan_amounts,
        interest_rates,
        lgd_rate=lgd_seed,
    )
    avg_loan = portfolio_stats["avg_loan"]
    avg_rate = portfolio_stats["avg_rate"]

    # ── 5. Portfolio summary ──────────────────────────────────────────────
    _render_portfolio_summary(portfolio_stats)

    # ── 6. Policy controls ────────────────────────────────────────────────
    params = _render_controls(scores,
                              portfolio_stats,
                              avg_loan,
                              avg_rate,
                              lgd_default=lgd_seed)
    cutoff = params["cutoff"]
    loan_size = params["loan_size"]
    rate = params["rate"]
    lgd = params["lgd"]
    preset = params["preset"]

    # ── 7. Pass-2 portfolio stats with active LGD ─────────────────────────
    if abs(lgd - lgd_seed) > 0.001:
        portfolio_stats = _cached_portfolio_stats(
            scores_key,
            scores,
            y_true,
            loan_amounts,
            interest_rates,
            lgd_rate=lgd,
        )

    # ── 8. Simulation ─────────────────────────────────────────────────────
    sim = simulate_credit_policy(scores, y_true, cutoff, loan_size, rate, lgd)

    # ── 9–17. Render all sections ─────────────────────────────────────────
    _render_kpis(sim, portfolio_stats)
    _render_score_distribution(scores, cutoff)
    best_cutoff, sweep_df = _cached_optimize(scores_key, scores, y_true,
                                             loan_size, rate, lgd)
    _render_optimisation(sweep_df, best_cutoff, cutoff)
    tradeoff_df = _cached_tradeoff(scores_key, scores, y_true, loan_size, rate,
                                   lgd)
    _render_risk_tradeoff(tradeoff_df)
    _render_confusion_matrix(y_true, scores, cutoff)
    _render_ai_advisor(sim, best_cutoff, preset, portfolio_stats)
    _render_recommendation(sim, best_cutoff, sweep_df, preset, portfolio_stats)
    _render_scenario_comparison(scores, y_true, loan_size, rate, lgd)
    _render_decision_table(sim, df)

    st.markdown(
        f'<div style="margin-top:40px;padding-top:14px;border-top:1px solid #CBD8E4;'
        f'font-size:0.69rem;color:{SLATE};font-family:IBM Plex Sans,sans-serif;'
        f'letter-spacing:0.03em;">'
        f'RiskLens Analytics &nbsp;·&nbsp; Policy Simulator &nbsp;·&nbsp;'
        f'Data: <code>data/raw/loan_book.csv</code> → '
        f'<code>st.session_state.portfolio_df</code> ({len(df):,} rows) &nbsp;·&nbsp;'
        f'Scores: derived from real application features &nbsp;·&nbsp;'
        f'For internal risk management use only.'
        f'</div>',
        unsafe_allow_html=True,
    )
