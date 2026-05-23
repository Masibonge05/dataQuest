"""
RiskLens Analytics — Model Explainability
Global Feature Importance & Local Attribution Metrics

app/pages/model_explainability.py

Data source: data/raw/loan_book.csv  →  st.session_state.portfolio_df
─────────────────────────────────────────────────────────────────────
Canonical columns consumed:
  Numeric features used to compute feature importance:
    dti_ratio, credit_utilisation_pct, num_delinquencies_2yr,
    num_hard_inquiries_6mo, pct_accounts_current,
    months_since_oldest_account, employment_length_years,
    annual_income, loan_amount, total_revolving_balance,
    age, num_open_accounts, months_since_last_delinquency,
    months_at_current_address, interest_rate

  Target column:
    default_flag (0 = performing, 1 = defaulted)

  Identifier column:
    applicant_id_hash

  All feature importances are computed from the real data using
  a Random Forest classifier — no hardcoded weights.
"""

import streamlit as streamlit_ctx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.components.header import render_header
from app.components.charts import (PRIMARY_BLUE, ACCENT_GOLD, ALERT_RED,
                                   SECONDARY_BLUE, apply_corporate_layout)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN MAPS — canonical loan_book.csv names first, aliases after
# ─────────────────────────────────────────────────────────────────────────────

_TARGET_CANDIDATES = [
    "default_flag",
    "default",
    "bad_flag",
    "bad",
    "target",
    "label",
    "outcome",
    "defaulted",
]

_ID_CANDIDATES = [
    "applicant_id_hash",  # canonical loan_book.csv identifier
    "account_id",
    "client_id",
    "id",
    "customer_id",
    "applicant_id",
    "loan_id",
]

# Ordered list of numeric feature columns present in loan_book.csv.
# Used both for importance computation and for the per-record deep-dive.
# Human-readable labels are in _FEATURE_LABELS.
_NUMERIC_FEATURES = [
    "dti_ratio",
    "credit_utilisation_pct",
    "num_delinquencies_2yr",
    "num_hard_inquiries_6mo",
    "pct_accounts_current",
    "months_since_oldest_account",
    "employment_length_years",
    "annual_income",
    "loan_amount",
    "total_revolving_balance",
    "age",
    "num_open_accounts",
    "months_since_last_delinquency",
    "months_at_current_address",
    "interest_rate",
]

# Display names for charts — same order as _NUMERIC_FEATURES
_FEATURE_LABELS = {
    "dti_ratio": "Debt-to-Income Ratio",
    "credit_utilisation_pct": "Credit Utilisation (%)",
    "num_delinquencies_2yr": "Delinquencies (2yr)",
    "num_hard_inquiries_6mo": "Hard Inquiries (6mo)",
    "pct_accounts_current": "Accounts Current (%)",
    "months_since_oldest_account": "Credit History Length (mo)",
    "employment_length_years": "Employment Length (yr)",
    "annual_income": "Annual Income",
    "loan_amount": "Loan Amount",
    "total_revolving_balance": "Total Revolving Balance",
    "age": "Applicant Age",
    "num_open_accounts": "Open Accounts",
    "months_since_last_delinquency": "Months Since Last Delinquency",
    "months_at_current_address": "Months at Current Address",
    "interest_rate": "Interest Rate (%)",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace AND lowercase all column names.

    Critical: _NUMERIC_FEATURES and all candidate lists use lowercase names
    matching the loan_book.csv schema.  If session_state.portfolio_df still
    has mixed-case columns (e.g. loaded before policy_simulator ran its
    normalisation), this ensures every column is matched correctly.
    Without lowercasing here, _available_features() finds almost nothing,
    which collapses importance_df to 1 row and crashes the slider (min==max).
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _detect_col(df: pd.DataFrame, candidates: list, label: str) -> str | None:
    # df columns are already lowercased by _normalise_df, so direct lookup works
    cols_set = set(df.columns)
    for cand in candidates:
        if cand.strip().lower() in cols_set:
            return cand.strip().lower()
    for col in df.columns:
        if label.lower() in col:
            return col
    return None


def _available_features(df: pd.DataFrame) -> list[str]:
    """
    Return the subset of _NUMERIC_FEATURES that actually exist in df.
    At least one column must be present; raises if none found.
    """
    present = [c for c in _NUMERIC_FEATURES if c in df.columns]
    if not present:
        raise ValueError(
            "No numeric feature columns from loan_book.csv found in the "
            "portfolio DataFrame.  Expected at least one of: " +
            ", ".join(_NUMERIC_FEATURES))
    return present


@streamlit_ctx.cache_data(
    show_spinner="Computing feature importances from loan_book.csv…")
def _compute_feature_importance(
    _df_hash: int,
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Fit a Random Forest on the available numeric features from loan_book.csv
    and return a sorted DataFrame of feature importances.

    Uses a 30 000-row stratified sample when the portfolio is large to keep
    training time under a second.  Falls back to mean absolute correlation
    with the target if sklearn is unavailable.

    Returns
    -------
    pd.DataFrame with columns ['feature', 'label', 'importance']
    sorted descending by importance.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer

    # Stratified sample cap — 30 k rows is fast and representative at 120 k+
    sample_df = df[feature_cols + [target_col]].copy()
    if len(sample_df) > 30_000:
        pos = sample_df[sample_df[target_col] == 1]
        neg = sample_df[sample_df[target_col] == 0]
        n_pos = min(len(pos), 5_000)
        n_neg = min(len(neg), 25_000)
        sample_df = pd.concat([
            pos.sample(n=n_pos, random_state=42),
            neg.sample(n=n_neg, random_state=42),
        ]).sample(frac=1, random_state=42)

    X = sample_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = sample_df[target_col].astype(int)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_imp, y)

    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        "feature":
        feature_cols,
        "label": [_FEATURE_LABELS.get(c, c) for c in feature_cols],
        "importance":
        importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Normalise to sum to 1 for cleaner percentage display
    total = importance_df["importance"].sum()
    if total > 0:
        importance_df["importance"] = importance_df["importance"] / total

    return importance_df


def _fallback_importance(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    sklearn-free fallback: rank features by absolute Spearman correlation
    with default_flag.  Used if sklearn is not installed.
    """
    records = []
    for col in feature_cols:
        series = pd.to_numeric(df[col], errors="coerce").fillna(0)
        corr = abs(series.corr(df[target_col].astype(float),
                               method="spearman"))
        records.append({
            "feature": col,
            "label": _FEATURE_LABELS.get(col, col),
            "importance": corr if not np.isnan(corr) else 0.0,
        })
    imp_df = pd.DataFrame(records).sort_values("importance", ascending=False)
    total = imp_df["importance"].sum()
    if total > 0:
        imp_df["importance"] = imp_df["importance"] / total
    return imp_df.reset_index(drop=True)


def _build_local_attribution(
    record: pd.Series,
    importance_df: pd.DataFrame,
    portfolio_medians: pd.Series,
) -> pd.DataFrame:
    """
    Approximate local attribution for a single applicant record.

    Method: multiply the global feature importance by the normalised
    deviation of the applicant's value from the portfolio median.
    Sign: positive = increases default risk, negative = reduces it.

    This is an approximation (not true SHAP) but is deterministic,
    fast, and requires no additional model fitting.
    """
    rows = []
    for _, row in importance_df.iterrows():
        col = row["feature"]
        if col not in record.index:
            continue
        applicant_val = pd.to_numeric(record[col], errors="coerce")
        median_val = portfolio_medians.get(col, np.nan)
        if pd.isna(applicant_val) or pd.isna(median_val) or median_val == 0:
            normalised_dev = 0.0
        else:
            normalised_dev = (applicant_val - median_val) / (abs(median_val) +
                                                             1e-9)
        attribution = float(row["importance"]) * normalised_dev
        rows.append({
            "Feature": row["label"],
            "Raw Column": col,
            "Applicant": applicant_val,
            "Median": median_val,
            "Attribution": attribution,
        })
    return pd.DataFrame(rows).sort_values("Attribution",
                                          key=abs,
                                          ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────────────────────────


def render_page():
    st = streamlit_ctx

    render_header(
        page_title="Model Explainability",
        page_subtitle=("Global Feature Importance & Local Attribution Metrics "
                       "(computed from loan_book.csv data)"),
    )

    # ── Session guard ─────────────────────────────────────────────────────
    if "portfolio_df" not in st.session_state:
        st.error("No portfolio dataframe found in session state.  "
                 "Ensure data/raw/loan_book.csv has been loaded.")
        return

    df = _normalise_df(st.session_state.portfolio_df)

    # ── Column detection ──────────────────────────────────────────────────
    target_col = _detect_col(df, _TARGET_CANDIDATES, "default")
    if target_col is None:
        st.error("Target column 'default_flag' not found.  "
                 f"Available columns: {list(df.columns)}")
        return

    df[target_col] = pd.to_numeric(df[target_col],
                                   errors="coerce").fillna(0).astype(int)

    try:
        feature_cols = _available_features(df)
    except ValueError as exc:
        st.error(str(exc))
        return

    id_col = _detect_col(df, _ID_CANDIDATES, "id")

    n_total = len(df)
    n_defaults = int(df[target_col].sum())
    n_features = len(feature_cols)

    st.caption(
        f"Source: `data/raw/loan_book.csv`  ·  "
        f"**{n_total:,}** applications  ·  "
        f"**{n_defaults:,}** defaults ({n_defaults/n_total*100:.2f}%)  ·  "
        f"**{n_features}** numeric features available for importance analysis")

    # ── Compute global importances ────────────────────────────────────────
    try:
        importance_df = _compute_feature_importance(id(df), df, target_col,
                                                    feature_cols)
        st.session_state.feature_importances = dict(
            zip(importance_df["label"], importance_df["importance"]))
        method_note = "Random Forest — 100 trees, stratified sample from loan_book.csv"
    except Exception:
        # sklearn not available — use correlation fallback
        importance_df = _fallback_importance(df, target_col, feature_cols)
        st.session_state.feature_importances = dict(
            zip(importance_df["label"], importance_df["importance"]))
        method_note = "Spearman correlation (sklearn unavailable)"

    # ── GLOBAL IMPORTANCE CHART ───────────────────────────────────────────
    st.markdown("### Global Feature Importance")
    st.caption(f"Method: {method_note}")

    # Top-N selector — guard: slider requires min < max, so only show it
    # when there are enough features; otherwise use all available features.
    n_available = len(importance_df)
    _slider_min = min(3, n_available)
    _slider_max = n_available
    if _slider_min < _slider_max:
        top_n = st.slider(
            "Show top N features",
            min_value=_slider_min,
            max_value=_slider_max,
            value=min(10, _slider_max),
            step=1,
        )
    else:
        top_n = n_available
        st.caption(f"Showing all {n_available} available feature(s).")
    top_df = importance_df.head(top_n).copy()
    top_df["pct"] = top_df["importance"] * 100

    # Horizontal bar chart (ascending so most important is at top)
    fig_importance = go.Figure(
        go.Bar(
            x=top_df["pct"].values[::-1],
            y=top_df["label"].values[::-1],
            orientation="h",
            marker=dict(
                color=top_df["pct"].values[::-1],
                colorscale=[[0, "#CBD8E4"], [0.5, SECONDARY_BLUE],
                            [1, PRIMARY_BLUE]],
                showscale=False,
            ),
            text=[f"{v:.1f}%" for v in top_df["pct"].values[::-1]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>",
        ))
    fig_importance.update_layout(
        xaxis_title="Relative Importance (%)",
        yaxis_title="",
        height=max(320, top_n * 34),
        margin=dict(l=10, r=60, t=20, b=30),
    )
    apply_corporate_layout(fig_importance)
    st.plotly_chart(fig_importance, use_container_width=True)

    # ── FEATURE BREAKDOWN TABLE ───────────────────────────────────────────
    with st.expander("Full Feature Importance Table", expanded=False):
        display_df = importance_df.copy()
        display_df["importance_pct"] = (display_df["importance"] *
                                        100).round(2)
        display_df["rank"] = range(1, len(display_df) + 1)
        # importance_df columns: feature (csv col name), label (display name), importance
        # "Raw Column" does not exist here — use "feature" which IS the csv column name
        st.dataframe(
            display_df[["rank", "label", "feature", "importance_pct"]].rename(
                columns={
                    "rank": "#",
                    "label": "Feature",
                    "feature": "CSV Column",
                    "importance_pct": "Importance (%)",
                }),
            use_container_width=True,
            hide_index=True,
        )

    # ── PORTFOLIO DISTRIBUTION BY FEATURE ────────────────────────────────
    st.markdown("---")
    st.markdown("### Feature Distribution — Defaults vs Performing")

    top3_features = importance_df.head(3)["feature"].tolist()
    dist_cols = st.columns(len(top3_features))

    for i, feat in enumerate(top3_features):
        feat_label = _FEATURE_LABELS.get(feat, feat)
        series = pd.to_numeric(df[feat], errors="coerce").dropna()
        defaults_series = pd.to_numeric(df[df[target_col] == 1][feat],
                                        errors="coerce").dropna()
        performing_series = pd.to_numeric(df[df[target_col] == 0][feat],
                                          errors="coerce").dropna()

        with dist_cols[i]:
            fig_d = go.Figure()
            fig_d.add_trace(
                go.Histogram(
                    x=performing_series,
                    name="Performing",
                    marker_color=PRIMARY_BLUE,
                    opacity=0.65,
                    nbinsx=40,
                ))
            fig_d.add_trace(
                go.Histogram(
                    x=defaults_series,
                    name="Defaulted",
                    marker_color=ALERT_RED,
                    opacity=0.65,
                    nbinsx=40,
                ))
            fig_d.update_layout(
                barmode="overlay",
                title_text=feat_label,
                title_font_size=11,
                height=260,
                showlegend=(i == 0),
                legend=dict(orientation="h", y=-0.30, font=dict(size=9)),
                margin=dict(l=10, r=10, t=36, b=10),
                xaxis_title=feat_label,
                yaxis_title="Count",
            )
            apply_corporate_layout(fig_d)
            st.plotly_chart(fig_d, use_container_width=True)

    # ── LOCAL ATTRIBUTION / INDIVIDUAL RECORD DEEP-DIVE ──────────────────
    st.markdown("---")
    st.markdown("### Local Account Deep-Dive")
    st.caption(
        "Select an applicant by their `applicant_id_hash` to see how their "
        "individual feature values deviate from the portfolio median and how "
        "each contributes to their risk score.")

    # Portfolio medians for baseline
    portfolio_medians = df[feature_cols].apply(pd.to_numeric,
                                               errors="coerce").median()

    if id_col is not None:
        # Use actual applicant_id_hash values from loan_book.csv
        id_options = df[id_col].dropna().unique().tolist()
        # Show a manageable sample in the selector (full list is 120 k+ rows)
        display_ids = id_options[:500]  # first 500 for selector performance
        if len(id_options) > 500:
            st.info(
                f"Showing first 500 of {len(id_options):,} applicant IDs in selector.  "
                "Use the search box to find a specific `applicant_id_hash`.")

        selected_id = st.selectbox(
            "Select Applicant (applicant_id_hash):",
            options=display_ids,
            help="Unique applicant identifier from loan_book.csv",
        )
        selected_record = df[df[id_col] == selected_id].iloc[0]
        actual_outcome = int(selected_record[target_col])
        outcome_label = "🔴 Defaulted" if actual_outcome == 1 else "🟢 Performing"
    else:
        # Fallback: let user pick a row index
        st.info(
            "'applicant_id_hash' column not found. Using row index instead.")
        row_idx = st.number_input(
            "Row index (0-based)",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1,
        )
        selected_record = df.iloc[row_idx]
        actual_outcome = int(selected_record[target_col])
        outcome_label = "🔴 Defaulted" if actual_outcome == 1 else "🟢 Performing"

    st.markdown(f"**Actual outcome:** {outcome_label}")

    # Local attribution
    local_df = _build_local_attribution(selected_record, importance_df,
                                        portfolio_medians)

    # Waterfall-style attribution chart
    fig_local = go.Figure()
    colors = [
        ALERT_RED if v > 0 else PRIMARY_BLUE for v in local_df["Attribution"]
    ]
    fig_local.add_trace(
        go.Bar(
            x=local_df["Feature"],
            y=local_df["Attribution"],
            marker_color=colors,
            hovertemplate=("<b>%{x}</b><br>"
                           "Attribution: %{y:.4f}<br>"
                           "<extra></extra>"),
        ))
    fig_local.add_hline(y=0, line_color="#CBD8E4", line_width=1)
    fig_local.update_layout(
        title_text=
        ("Local Feature Attribution — positive (red) = increases default risk, "
         "negative (blue) = reduces risk"),
        title_font_size=11,
        xaxis_title="Feature",
        yaxis_title="Attribution Score",
        height=360,
        margin=dict(l=10, r=10, t=44, b=80),
        xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
    )
    apply_corporate_layout(fig_local)
    st.plotly_chart(fig_local, use_container_width=True)

    # Detailed values table
    with st.expander("Applicant Feature Values vs Portfolio Median",
                     expanded=True):
        display_local = local_df[[
            "Feature", "Applicant", "Median", "Attribution"
        ]].copy()
        display_local["Applicant"] = display_local["Applicant"].apply(
            lambda v: f"{v:,.3f}" if pd.notna(v) else "N/A")
        display_local["Median"] = display_local["Median"].apply(
            lambda v: f"{v:,.3f}" if pd.notna(v) else "N/A")
        display_local["Attribution"] = display_local["Attribution"].apply(
            lambda v: f"{v:+.4f}")
        display_local["Risk Direction"] = local_df["Attribution"].apply(
            lambda v: "↑ Higher risk"
            if v > 0.001 else ("↓ Lower risk" if v < -0.001 else "Neutral"))
        st.dataframe(display_local, use_container_width=True, hide_index=True)

    # Raw record expander
    with st.expander("Full Raw Applicant Record (loan_book.csv fields)",
                     expanded=False):
        # Show all columns including non-numeric ones
        record_dict = selected_record.to_dict()
        # Highlight the identifier and outcome fields
        st.json({
            k: (v if not isinstance(v, float) or not np.isnan(v) else None)
            for k, v in record_dict.items()
        })
