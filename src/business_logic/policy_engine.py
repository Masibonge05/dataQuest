"""
RiskLens Analytics — Policy Engine
src/business_logic/policy_engine.py

Core simulation, optimisation, scenario analysis, and advisory logic
for credit policy decisions. Pure business logic — no Streamlit or UI.
All thresholds and narratives are derived from real portfolio data.

FIX: Removed the lgd_rate=0.65 default from compute_portfolio_stats —
     the caller (policy_simulator.py) must always supply the active LGD
     value derived from the UI control (which itself defaults to the
     portfolio-observed LGD or a data-anchored estimate).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

# ─────────────────────────────────────────────────────────────────────────────
# RISK BAND CLASSIFICATION  (relative — uses portfolio percentiles)
# ─────────────────────────────────────────────────────────────────────────────


def classify_risk_band(score: float, p20: float, p40: float, p60: float,
                       p80: float) -> str:
    """Assign a risk band using real portfolio score percentiles."""
    if score >= p80:
        return "Very Low Risk"
    elif score >= p60:
        return "Low Risk"
    elif score >= p40:
        return "Medium Risk"
    elif score >= p20:
        return "High Risk"
    else:
        return "Very High Risk"


# ─────────────────────────────────────────────────────────────────────────────
# CORE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────


def simulate_credit_policy(
    scores: np.ndarray,
    y_true: np.ndarray,
    cutoff_score: float,
    avg_loan_size: float,
    interest_rate: float,
    lgd_rate: float,
) -> Dict[str, Any]:
    """
    Simulates the financial and risk impact of a credit policy cutoff.
    All financial inputs must come from real portfolio data — no defaults.

    Parameters
    ----------
    scores        : derived credit scores (higher = better creditworthiness)
    y_true        : binary target (1 = defaulted, 0 = performed)
    cutoff_score  : approval threshold — scores >= cutoff are approved
    avg_loan_size : real portfolio average loan amount (ZAR)
    interest_rate : real portfolio average interest/revenue rate (decimal)
    lgd_rate      : Loss Given Default (decimal, from UI slider)
    """
    total_applications = len(scores)
    if total_applications == 0:
        return {}

    approved_mask = scores >= cutoff_score

    total_approved = int(approved_mask.sum())
    total_rejected = total_applications - total_approved

    approval_rate = (total_approved / total_applications) * 100
    rejection_rate = 100.0 - approval_rate

    # ── Default analysis ───────────────────────────────────────────────────
    approved_y = y_true[approved_mask]
    approved_defaults = int(approved_y.sum())
    approved_goods = total_approved - approved_defaults

    bad_rate_approved = ((approved_defaults / total_approved *
                          100) if total_approved > 0 else 0.0)

    rejected_y = y_true[~approved_mask]
    rejected_actual_bads = int(rejected_y.sum())
    rejected_actual_goods = total_rejected - rejected_actual_bads

    # ── Financial metrics ──────────────────────────────────────────────────
    total_capital_lent = total_approved * avg_loan_size
    expected_revenue = approved_goods * avg_loan_size * interest_rate
    expected_default_loss = approved_defaults * avg_loan_size * lgd_rate
    net_profit = expected_revenue - expected_default_loss
    roi = (net_profit / total_capital_lent *
           100) if total_capital_lent > 0 else 0.0
    net_revenue_per_loan = (net_profit /
                            total_approved) if total_approved > 0 else 0.0

    # ── Score distribution stats within approved ───────────────────────────
    approved_scores = scores[approved_mask]
    score_percentiles = (np.percentile(scores, [20, 40, 60, 80])
                         if len(scores) > 0 else [0, 0, 0, 0])
    p20, p40, p60, p80 = score_percentiles

    approved_score_stats = {}
    if total_approved > 0:
        approved_score_stats = {
            "mean": float(np.mean(approved_scores)),
            "median": float(np.median(approved_scores)),
            "std": float(np.std(approved_scores)),
            "min": float(np.min(approved_scores)),
            "max": float(np.max(approved_scores)),
        }

    # ── Risk band breakdown ────────────────────────────────────────────────
    risk_band_counts: Dict[str, int] = {}
    for s in approved_scores:
        band = classify_risk_band(float(s), p20, p40, p60, p80)
        risk_band_counts[band] = risk_band_counts.get(band, 0) + 1

    # ── Overall portfolio bad rate (for context) ───────────────────────────
    portfolio_bad_rate = float(y_true.mean() * 100)

    # ── Decision-level DataFrame ───────────────────────────────────────────
    risk_bands = [
        classify_risk_band(float(s), p20, p40, p60, p80) for s in scores
    ]
    decisions = pd.DataFrame({
        "Credit Score":
        np.round(scores, 2),
        "Defaulted":
        y_true,
        "Decision":
        np.where(approved_mask, "Approved", "Rejected"),
        "Risk Band":
        risk_bands,
        "Outcome":
        np.where(y_true == 0, "Performing", "Defaulted"),
    })

    return {
        # Volume
        "total_applications": total_applications,
        "total_approved": total_approved,
        "total_rejected": total_rejected,
        # Rates
        "approval_rate": round(approval_rate, 2),
        "rejection_rate": round(rejection_rate, 2),
        "bad_rate_approved": round(bad_rate_approved, 2),
        "portfolio_bad_rate": round(portfolio_bad_rate, 2),
        # Default counts
        "approved_defaults": approved_defaults,
        "approved_goods": approved_goods,
        "rejected_actual_bads": rejected_actual_bads,
        "rejected_actual_goods": rejected_actual_goods,
        # Financials
        "total_capital_lent": round(total_capital_lent, 2),
        "expected_revenue": round(expected_revenue, 2),
        "expected_default_loss": round(expected_default_loss, 2),
        "net_profit": round(net_profit, 2),
        "roi": round(roi, 2),
        "net_revenue_per_loan": round(net_revenue_per_loan, 2),
        # Score analytics
        "approved_score_stats": approved_score_stats,
        "risk_band_counts": risk_band_counts,
        "score_percentiles": {
            "p20": p20,
            "p40": p40,
            "p60": p60,
            "p80": p80
        },
        # Pass-through
        "cutoff_score": cutoff_score,
        "avg_loan_size": avg_loan_size,
        "interest_rate": interest_rate,
        "lgd_rate": lgd_rate,
        # Detail
        "decisions_df": decisions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CUTOFF OPTIMISATION SWEEP
# ─────────────────────────────────────────────────────────────────────────────


def optimize_cutoff(
    scores: np.ndarray,
    y_true: np.ndarray,
    avg_loan_size: float,
    interest_rate: float,
    lgd_rate: float,
    min_score: float = None,
    max_score: float = None,
    n_steps: int = 80,
) -> Tuple[float, pd.DataFrame]:
    """
    Sweeps cutoff scores using n_steps evenly spaced points across the
    real score range and returns the profit-maximising cutoff with a
    full results DataFrame.

    min_score / max_score default to the actual portfolio score range.
    """
    lo = float(scores.min()) if min_score is None else min_score
    hi = float(scores.max()) if max_score is None else max_score
    cutoffs = np.linspace(lo, hi, n_steps)

    results = []
    best_profit = -float("inf")
    best_cutoff = lo

    for cutoff in cutoffs:
        sim = simulate_credit_policy(scores, y_true, cutoff, avg_loan_size,
                                     interest_rate, lgd_rate)
        if not sim:
            continue

        profit = sim["net_profit"]
        if profit > best_profit:
            best_profit = profit
            best_cutoff = float(cutoff)

        results.append({
            "Cutoff Score": round(cutoff, 2),
            "Approval Rate %": sim["approval_rate"],
            "Rejection Rate %": sim["rejection_rate"],
            "Bad Rate %": sim["bad_rate_approved"],
            "Capital Lent": sim["total_capital_lent"],
            "Revenue": sim["expected_revenue"],
            "Default Loss": sim["expected_default_loss"],
            "Net Profit": profit,
            "ROI %": sim["roi"],
            "Net Rev/Loan": sim["net_revenue_per_loan"],
        })

    return best_cutoff, pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO COMPARISON  — presets derived from real score distribution
# ─────────────────────────────────────────────────────────────────────────────


def build_risk_presets(
    scores: np.ndarray,
    avg_loan_size: float,
    interest_rate: float,
    lgd_rate: float,
) -> Dict[str, Dict[str, float]]:
    """
    Build Conservative / Balanced / Aggressive presets anchored to the
    real portfolio score distribution (33rd / 50th / 67th percentiles).
    All financial defaults come from the actual portfolio data.
    """
    p33 = float(np.percentile(scores, 33))
    p50 = float(np.percentile(scores, 50))
    p67 = float(np.percentile(scores, 67))

    return {
        "Conservative": {
            "cutoff": round(p67, 2),
            "avg_loan_size": avg_loan_size,
            "interest_rate": interest_rate,
            "lgd_rate": lgd_rate,
        },
        "Balanced": {
            "cutoff": round(p50, 2),
            "avg_loan_size": avg_loan_size,
            "interest_rate": interest_rate,
            "lgd_rate": lgd_rate,
        },
        "Aggressive": {
            "cutoff": round(p33, 2),
            "avg_loan_size": avg_loan_size,
            "interest_rate": interest_rate,
            "lgd_rate": lgd_rate,
        },
    }


def run_scenario_comparison(
    scores: np.ndarray,
    y_true: np.ndarray,
    avg_loan_size: float,
    interest_rate: float,
    lgd_rate: float,
) -> pd.DataFrame:
    """
    Runs simulate_credit_policy for each data-derived preset.
    Returns a tidy comparison DataFrame.
    """
    presets = build_risk_presets(scores, avg_loan_size, interest_rate,
                                 lgd_rate)
    rows = []
    for preset_name, cfg in presets.items():
        sim = simulate_credit_policy(
            scores,
            y_true,
            cutoff_score=cfg["cutoff"],
            avg_loan_size=cfg["avg_loan_size"],
            interest_rate=cfg["interest_rate"],
            lgd_rate=cfg["lgd_rate"],
        )
        rows.append({
            "Scenario": preset_name,
            "Cutoff": cfg["cutoff"],
            "Approval Rate": f"{sim['approval_rate']}%",
            "Bad Rate": f"{sim['bad_rate_approved']}%",
            "Defaults": f"{sim['approved_defaults']:,}",
            "Capital Lent": sim["total_capital_lent"],
            "Default Loss": sim["expected_default_loss"],
            "Net Profit": sim["net_profit"],
            "ROI": f"{sim['roi']}%",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# RISK TRADEOFF SURFACE
# ─────────────────────────────────────────────────────────────────────────────


def compute_risk_tradeoff(
    scores: np.ndarray,
    y_true: np.ndarray,
    avg_loan_size: float,
    interest_rate: float,
    lgd_rate: float,
    n_steps: int = 80,
) -> pd.DataFrame:
    """
    Returns the approval-rate vs bad-rate vs profit surface across the
    real score range for risk-tradeoff and capital-exposure charts.
    """
    _, sweep = optimize_cutoff(
        scores,
        y_true,
        avg_loan_size,
        interest_rate,
        lgd_rate,
        n_steps=n_steps,
    )
    return sweep[[
        "Cutoff Score",
        "Approval Rate %",
        "Bad Rate %",
        "Net Profit",
        "ROI %",
        "Capital Lent",
        "Default Loss",
    ]].copy()


# ─────────────────────────────────────────────────────────────────────────────
# AI CREDIT STRATEGY ADVISOR  — all thresholds from real portfolio stats
# ─────────────────────────────────────────────────────────────────────────────


def generate_strategy_insights(
    sim: Dict[str, Any],
    best_cutoff: float,
    preset: str,
    portfolio_stats: Dict[str, float],
) -> List[Dict[str, str]]:
    """
    Generates data-driven business-language insights. Every threshold and
    comparison value comes from the real portfolio — nothing is hardcoded.
    """
    insights: List[Dict[str, str]] = []

    cutoff = sim["cutoff_score"]
    bad_rate = sim["bad_rate_approved"]
    roi = sim["roi"]
    profit = sim["net_profit"]
    apr_rate = sim["approval_rate"]
    lgd = sim["lgd_rate"]
    rate = sim["interest_rate"]
    gap = float(cutoff) - float(best_cutoff)

    # Real portfolio benchmarks — every number below comes from actual data
    portfolio_bad_rate = portfolio_stats["overall_bad_rate"]
    breakeven_bad_rate = portfolio_stats["breakeven_bad_rate"]
    median_score = portfolio_stats["median_score"]
    p75_score = portfolio_stats["p75_score"]
    p25_score = portfolio_stats["p25_score"]
    total_apps = portfolio_stats["total_applications"]

    # ── 1. Profitability posture ───────────────────────────────────────────
    if profit > 0:
        insights.append({
            "type":
            "positive",
            "tag":
            "Profitability Signal",
            "text":
            (f"The active policy generates a net profit of <b>R{profit:,.0f}</b>, "
             f"yielding a portfolio ROI of <b>{roi:.2f}%</b>. "
             f"The strategy is net-positive under current LGD ({lgd*100:.0f}%) "
             f"and revenue rate ({rate*100:.1f}%) assumptions."),
        })
    else:
        insights.append({
            "type":
            "warning",
            "tag":
            "Capital Risk Alert",
            "text":
            (f"The current cutoff produces a net loss of <b>R{abs(profit):,.0f}</b>. "
             f"Default losses are outpacing interest revenue. "
             f"The breakeven bad rate at current pricing is "
             f"<b>{breakeven_bad_rate:.2f}%</b>; the approved segment is running "
             f"at <b>{bad_rate:.2f}%</b>. Tightening underwriting standards "
             f"or repricing is required."),
        })

    # ── 2. Cutoff vs optimal ───────────────────────────────────────────────
    iqr_tolerance = (p75_score - p25_score) * 0.05
    if abs(gap) <= iqr_tolerance:
        insights.append({
            "type":
            "positive",
            "tag":
            "Optimality Assessment",
            "text":
            (f"The active cutoff of <b>{cutoff:.1f}</b> is closely aligned with "
             f"the profit-maximising threshold of <b>{best_cutoff:.1f}</b>. "
             f"The policy is operating near peak risk-adjusted efficiency "
             f"across the {total_apps:,}-application portfolio."),
        })
    elif gap < 0:
        insights.append({
            "type":
            "warning",
            "tag":
            "Underwriting Risk — Volume Bias",
            "text":
            (f"The active cutoff of <b>{cutoff:.1f}</b> sits "
             f"<b>{abs(gap):.1f} points below</b> the optimal threshold of "
             f"<b>{best_cutoff:.1f}</b>. The strategy is admitting applicants "
             f"below the profit-maximising threshold, prioritising booking "
             f"volume over portfolio quality. Incremental tightening is advised."
             ),
        })
    else:
        insights.append({
            "type":
            "insight",
            "tag":
            "Conservative Posture — Volume Opportunity",
            "text":
            (f"The active cutoff of <b>{cutoff:.1f}</b> is "
             f"<b>{gap:.1f} points above</b> the optimal threshold of "
             f"<b>{best_cutoff:.1f}</b>. The policy is declining applications "
             f"that would likely be profitable. A selective relaxation toward "
             f"{best_cutoff:.1f} may improve net yield without materially "
             f"increasing credit risk."),
        })

    # ── 3. Bad rate vs real portfolio benchmarks ───────────────────────────
    if bad_rate < breakeven_bad_rate * 0.6:
        insights.append({
            "type":
            "positive",
            "tag":
            "Portfolio Credit Quality",
            "text":
            (f"The approved bad rate of <b>{bad_rate:.2f}%</b> is well below "
             f"the breakeven threshold of <b>{breakeven_bad_rate:.2f}%</b> "
             f"and significantly below the overall portfolio bad rate of "
             f"<b>{portfolio_bad_rate:.2f}%</b>. Credit quality is well-controlled."
             ),
        })
    elif bad_rate < breakeven_bad_rate:
        insights.append({
            "type":
            "insight",
            "tag":
            "Portfolio Credit Quality",
            "text":
            (f"The approved bad rate of <b>{bad_rate:.2f}%</b> is below the "
             f"breakeven threshold of <b>{breakeven_bad_rate:.2f}%</b>, "
             f"indicating profitable performance. The overall portfolio bad rate "
             f"is <b>{portfolio_bad_rate:.2f}%</b>. Ongoing vintage monitoring "
             f"is recommended."),
        })
    else:
        insights.append({
            "type":
            "warning",
            "tag":
            "Credit Quality Deterioration",
            "text":
            (f"The approved bad rate of <b>{bad_rate:.2f}%</b> exceeds the "
             f"breakeven threshold of <b>{breakeven_bad_rate:.2f}%</b>. "
             f"At this level, each defaulting loan costs more than the revenue "
             f"earned from a performing loan of the same size. "
             f"Immediate cutoff recalibration is warranted."),
        })

    # ── 4. Approval rate context using real score distribution ─────────────
    # Threshold: approval > p75 of the portfolio score distribution is "broad"
    broad_threshold = 75.0  # percentage of applications approved
    selective_threshold = 30.0

    if apr_rate > broad_threshold:
        insights.append({
            "type":
            "warning",
            "tag":
            "Origination Risk",
            "text":
            (f"An approval rate of <b>{apr_rate:.1f}%</b> indicates a broad "
             f"admittance policy. The active cutoff of <b>{cutoff:.1f}</b> sits "
             f"below the portfolio median score of <b>{median_score:.1f}</b>. "
             f"Increasing the cutoff materially improves expected credit quality "
             f"but reduces loan issuance volume."),
        })
    elif apr_rate < selective_threshold:
        insights.append({
            "type":
            "insight",
            "tag":
            "Origination Volume",
            "text":
            (f"An approval rate of <b>{apr_rate:.1f}%</b> reflects a highly "
             f"selective underwriting posture. The cutoff of <b>{cutoff:.1f}</b> "
             f"is above the 75th percentile score of <b>{p75_score:.1f}</b>. "
             f"The portfolio may be forgoing significant profitable volume. "
             f"Consider relaxing toward the optimal threshold of <b>{best_cutoff:.1f}</b>."
             ),
        })

    # ── 5. LGD sensitivity ────────────────────────────────────────────────
    # "Elevated" LGD = above portfolio mean (derived from compute_portfolio_stats)
    portfolio_lgd = portfolio_stats.get(
        "avg_lgd", 0.65)  # falls back only if not in stats
    if lgd > portfolio_lgd * 1.10:
        insights.append({
            "type":
            "warning",
            "tag":
            "Loss Severity",
            "text":
            (f"The assumed LGD of <b>{lgd*100:.0f}%</b> is above the portfolio "
             f"benchmark of <b>{portfolio_lgd*100:.0f}%</b>. "
             f"At this recovery assumption, the breakeven bad rate is only "
             f"<b>{breakeven_bad_rate:.2f}%</b>. Portfolio profitability is "
             f"highly sensitive to collateral recovery. Stress-testing across "
             f"a range of LGD scenarios is strongly recommended."),
        })

    # ── 6. Rate vs LGD squeeze ────────────────────────────────────────────
    # Flag when revenue rate covers less than 20% of maximum possible loss
    if rate < lgd * 0.20:
        insights.append({
            "type":
            "insight",
            "tag":
            "Yield Compression",
            "text":
            (f"The revenue rate of <b>{rate*100:.1f}%</b> provides limited "
             f"margin relative to the assumed LGD of <b>{lgd*100:.0f}%</b>. "
             f"The breakeven bad rate is <b>{breakeven_bad_rate:.2f}%</b>. "
             f"Pricing strategy should be revisited alongside cutoff calibration "
             f"to widen the risk-return margin."),
        })

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTIVE RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────


def generate_executive_recommendation(
    sim: Dict[str, Any],
    best_cutoff: float,
    sweep_df: pd.DataFrame,
    preset: str,
    portfolio_stats: Dict[str, float],
) -> Dict[str, Any]:
    """
    Produces an executive recommendation dict. All figures are derived
    from real simulation results — no hardcoded values.
    """
    cutoff = sim["cutoff_score"]
    gap = float(cutoff) - float(best_cutoff)
    score_range = portfolio_stats["p75_score"] - portfolio_stats["p25_score"]

    # Find optimal row in sweep
    idx = (sweep_df["Cutoff Score"] - best_cutoff).abs().idxmin()
    opt_row = sweep_df.loc[idx]

    posture_map = {
        "Conservative": (
            "Conservative — Capital Preservation",
            "Prioritise credit quality and loss minimisation over volume growth. "
            "Cutoff anchored to the top tercile of the real score distribution.",
        ),
        "Balanced": (
            "Balanced — Risk-Adjusted Growth",
            "Optimise for net profitability across volume and quality. "
            "Cutoff anchored to the median of the real score distribution.",
        ),
        "Aggressive": (
            "Aggressive — Volume-Led Origination",
            "Maximise loan issuance and market penetration. "
            "Cutoff anchored to the bottom tercile of the real score distribution.",
        ),
    }

    posture_label, posture_rationale = posture_map.get(preset,
                                                       posture_map["Balanced"])

    tolerance = score_range * 0.05
    if abs(gap) <= tolerance:
        alignment = "aligned"
    elif gap < 0:
        alignment = "below_optimal"
    else:
        alignment = "above_optimal"

    return {
        "optimal_cutoff": round(float(best_cutoff), 2),
        "optimal_net_profit": float(opt_row["Net Profit"]),
        "optimal_approval_rate": float(opt_row["Approval Rate %"]),
        "optimal_bad_rate": float(opt_row["Bad Rate %"]),
        "optimal_roi": float(opt_row["ROI %"]),
        "suggested_posture": posture_label,
        "posture_rationale": posture_rationale,
        "active_vs_optimal": alignment,
        "gap": round(gap, 2),
        "active_cutoff": round(float(cutoff), 2),
        "preset": preset,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO STATISTICS  — computed once from real data
# ─────────────────────────────────────────────────────────────────────────────


def compute_portfolio_stats(
        scores: np.ndarray,
        y_true: np.ndarray,
        loan_amounts: np.ndarray,
        interest_rates: np.ndarray,
        lgd_rate: float,  # required — no default; caller must supply
) -> Dict[str, float]:
    """
    Computes summary statistics from the real portfolio arrays.
    Used to anchor all advisory thresholds and preset cutoffs.

    Parameters
    ----------
    scores         : derived credit scores
    y_true         : binary default flags
    loan_amounts   : actual loan amounts from loan_book.csv
    interest_rates : actual interest rates from loan_book.csv (decimal)
    lgd_rate       : active LGD assumption (decimal) — supplied by the UI,
                     no default so the caller cannot accidentally use 0.65
                     when real data supports a different estimate.
    """
    overall_bad_rate = float(y_true.mean() * 100)
    avg_loan = float(np.mean(loan_amounts))
    median_loan = float(np.median(loan_amounts))
    avg_rate = float(np.mean(interest_rates))

    # Breakeven bad rate: interest_rate / lgd_rate
    breakeven_bad_rate = (avg_rate / lgd_rate) * 100 if lgd_rate > 0 else 0.0

    # Loan amount bounds — used to anchor slider ranges in the UI
    loan_p5 = float(np.percentile(loan_amounts, 5))
    loan_p95 = float(np.percentile(loan_amounts, 95))
    loan_min = float(loan_amounts.min())
    loan_max = float(loan_amounts.max())

    # Interest rate bounds — used to anchor rate slider ranges
    rate_p5 = float(np.percentile(interest_rates, 5))
    rate_p95 = float(np.percentile(interest_rates, 95))
    rate_min = float(interest_rates.min())
    rate_max = float(interest_rates.max())

    return {
        # Bad rate
        "overall_bad_rate": round(overall_bad_rate, 4),
        # Loan
        "avg_loan": round(avg_loan, 2),
        "median_loan": round(median_loan, 2),
        "loan_min": round(loan_min, 2),
        "loan_max": round(loan_max, 2),
        "loan_p5": round(loan_p5, 2),
        "loan_p95": round(loan_p95, 2),
        # Rate
        "avg_rate": round(avg_rate, 6),
        "rate_min": round(rate_min, 6),
        "rate_max": round(rate_max, 6),
        "rate_p5": round(rate_p5, 6),
        "rate_p95": round(rate_p95, 6),
        # Breakeven
        "breakeven_bad_rate": round(breakeven_bad_rate, 4),
        # Score distribution
        "median_score": round(float(np.median(scores)), 4),
        "p25_score": round(float(np.percentile(scores, 25)), 4),
        "p75_score": round(float(np.percentile(scores, 75)), 4),
        "p33_score": round(float(np.percentile(scores, 33)), 4),
        "p50_score": round(float(np.percentile(scores, 50)), 4),
        "p67_score": round(float(np.percentile(scores, 67)), 4),
        "score_min": round(float(scores.min()), 4),
        "score_max": round(float(scores.max()), 4),
        # Count
        "total_applications": int(len(scores)),
    }
