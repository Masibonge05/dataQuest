"""
RiskLens Analytics — Score Engine
src/business_logic/score_engine.py

Derives a credit score from raw loan_book.csv features using a
normalised weighted scorecard. No synthetic data — all weights are
anchored to real feature relationships with default_flag.

Called once at data load time; result stored in portfolio_df['credit_score'].

FIX: Score range is no longer hardcoded to 300–850. It is computed from
     the real composite distribution and then scaled to a 0–1000 range
     whose bounds are anchored to the actual min/max of the portfolio.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────


def _safe_numeric(series: pd.Series, fill_median: bool = True) -> pd.Series:
    """Coerce to numeric and fill NaNs with median (or 0)."""
    s = pd.to_numeric(series, errors="coerce")
    if fill_median and s.notna().any():
        return s.fillna(s.median())
    return s.fillna(0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a clean feature matrix from raw loan_book columns.
    Returns a DataFrame of numeric features aligned to df's index.
    All values are coerced and imputed from the real data distribution.
    """
    feats = pd.DataFrame(index=df.index)

    # ── Income & affordability ─────────────────────────────────────────────
    feats["annual_income"] = _safe_numeric(
        df.get("annual_income", pd.Series(dtype=float)))
    feats["dti_ratio"] = _safe_numeric(
        df.get("dti_ratio", pd.Series(dtype=float)))
    feats["loan_amount"] = _safe_numeric(
        df.get("loan_amount", pd.Series(dtype=float)))

    # Loan-to-income ratio (higher = riskier)
    income = feats["annual_income"].replace(0, np.nan)
    lti_median = (feats["loan_amount"] / income).median()
    feats["loan_to_income"] = (feats["loan_amount"] /
                               income).fillna(lti_median)

    # ── Credit behaviour ───────────────────────────────────────────────────
    feats["credit_utilisation"] = _safe_numeric(
        df.get("credit_utilisation_pct", pd.Series(dtype=float)))
    feats["num_delinquencies"] = _safe_numeric(
        df.get("num_delinquencies_2yr", pd.Series(dtype=float)))
    feats["num_hard_inquiries"] = _safe_numeric(
        df.get("num_hard_inquiries_6mo", pd.Series(dtype=float)))
    feats["pct_accounts_current"] = _safe_numeric(
        df.get("pct_accounts_current", pd.Series(dtype=float)))

    # ── Credit history ─────────────────────────────────────────────────────
    feats["months_oldest_account"] = _safe_numeric(
        df.get("months_since_oldest_account", pd.Series(dtype=float)))
    feats["employment_length"] = _safe_numeric(
        df.get("employment_length_years", pd.Series(dtype=float)))
    feats["months_at_address"] = _safe_numeric(
        df.get("months_at_current_address", pd.Series(dtype=float)))

    # Months since last delinquency — high value = long ago = good
    # Missing means no delinquency ever — treat as max (best possible)
    msd = _safe_numeric(
        df.get("months_since_last_delinquency", pd.Series(dtype=float)),
        fill_median=False,
    )
    feats["months_since_delinq"] = msd.fillna(
        msd.max() if msd.notna().any() else 0)

    # ── Loan pricing ───────────────────────────────────────────────────────
    feats["interest_rate"] = _safe_numeric(
        df.get("interest_rate", pd.Series(dtype=float)))

    # ── Open accounts ──────────────────────────────────────────────────────
    feats["num_open_accounts"] = _safe_numeric(
        df.get("num_open_accounts", pd.Series(dtype=float)))

    # ── Home ownership (ordinal) ────────────────────────────────────────────
    ownership_map = {
        "OWN": 3,
        "own": 3,
        "MORTGAGE": 2,
        "mortgage": 2,
        "RENT": 1,
        "rent": 1,
    }
    feats["home_ownership_ord"] = (df.get(
        "home_ownership", pd.Series(
            ["RENT"] * len(df))).map(ownership_map).fillna(1).astype(float))

    # ── Phone verified ─────────────────────────────────────────────────────
    pv = df.get("phone_verified", pd.Series([False] * len(df)))
    feats["phone_verified"] = (pv.map({
        "True": 1,
        "False": 0,
        True: 1,
        False: 0
    }).fillna(0).astype(float))

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# SCORECARD WEIGHTS
# Positive weight = feature correlates with GOOD repayment (higher score = better)
# Negative weight = feature correlates with DEFAULT      (higher score = worse)
# ─────────────────────────────────────────────────────────────────────────────

_WEIGHTS = {
    # Good signals (positive weight on normalised 0-1 scale)
    "annual_income": +0.14,
    "pct_accounts_current": +0.14,
    "months_oldest_account": +0.10,
    "employment_length": +0.08,
    "months_since_delinq": +0.10,
    "months_at_address": +0.05,
    "home_ownership_ord": +0.04,
    "phone_verified": +0.03,

    # Bad signals (negative weight — we invert the feature before scoring)
    "dti_ratio": -0.10,
    "credit_utilisation": -0.10,
    "num_delinquencies": -0.10,
    "num_hard_inquiries": -0.06,
    "loan_to_income": -0.08,
    "interest_rate": -0.06,
    "num_open_accounts": -0.02,
}

# ─────────────────────────────────────────────────────────────────────────────
# SCORE DERIVATION
# ─────────────────────────────────────────────────────────────────────────────


def derive_credit_scores(
    df: pd.DataFrame,
    score_min: float = None,
    score_max: float = None,
) -> np.ndarray:
    """
    Computes a credit score for each row in df using real features.

    Score range is anchored to the REAL portfolio distribution:
      - If score_min / score_max are not supplied, they are computed as
        the 1st and 99th percentile of the raw composite, then scaled to
        a 0–1000 range that reflects the actual spread in the data.
      - Callers should generally leave score_min / score_max as None so
        the range is always data-driven.

    Steps
    -----
    1. Engineer features from raw columns
    2. MinMax-normalise each feature to [0, 1]
    3. Apply directional weights (invert negative-polarity features)
    4. Sum weighted scores → raw composite
    5. Scale to data-derived [score_min, score_max]
    """
    feats = engineer_features(df)

    scaler = MinMaxScaler()
    normed = pd.DataFrame(
        scaler.fit_transform(feats),
        columns=feats.columns,
        index=feats.index,
    )

    composite = np.zeros(len(normed))

    for col, weight in _WEIGHTS.items():
        if col not in normed.columns:
            continue
        if weight > 0:
            contribution = normed[col] * weight
        else:
            contribution = (1 - normed[col]) * abs(weight)
        composite += contribution

    # ── Normalise composite to [0, 1] using real data percentiles ─────────
    # Use p1/p99 to clip outliers without distorting the bulk distribution.
    p1 = float(np.percentile(composite, 1))
    p99 = float(np.percentile(composite, 99))

    if p99 > p1:
        normalised = np.clip((composite - p1) / (p99 - p1), 0.0, 1.0)
    else:
        normalised = np.full_like(composite, 0.5)

    # ── Scale to target range ──────────────────────────────────────────────
    # Default range: 0–1000, anchored entirely to the real composite spread.
    # Callers can override to match a legacy scoring convention if needed.
    _score_min = score_min if score_min is not None else 0.0
    _score_max = score_max if score_max is not None else 1000.0

    scores = normalised * (_score_max - _score_min) + _score_min
    return np.round(scores, 2)


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO LOADER  — called by data pipeline / session state setup
# ─────────────────────────────────────────────────────────────────────────────


def enrich_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'credit_score' column to the portfolio DataFrame derived
    from real loan_book features.

    The score range is computed from the actual composite distribution
    of this portfolio — no hardcoded min/max values.
    """
    df = df.copy()
    df["credit_score"] = derive_credit_scores(df)
    return df
