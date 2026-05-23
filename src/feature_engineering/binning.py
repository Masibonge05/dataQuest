"""
RiskLens Analytics — WoE / IV Binning Engine
src/feature_engineering/binning.py
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def compute_woe_iv_table(df: pd.DataFrame, col: str, target_col: str,
                         bins: list) -> pd.DataFrame:
    temp = pd.DataFrame({'val': df[col], 'target': df[target_col]}).dropna()
    if len(temp) == 0:
        return pd.DataFrame()

    # Cast to float — prevents numpy boolean subtract error during interpolation
    temp['val'] = temp['val'].astype(float)
    temp['target'] = temp['target'].astype(float)

    temp['bin'] = pd.cut(temp['val'], bins=bins, include_lowest=True)
    total_goods = (temp['target'] == 0).sum()
    total_bads = (temp['target'] == 1).sum()
    if total_goods == 0 or total_bads == 0:
        return pd.DataFrame()

    grouped = temp.groupby('bin', observed=False).agg(
        count=('target', 'count'),
        bads=('target', lambda x: (x == 1).sum()),
        goods=('target', lambda x: (x == 0).sum()),
    ).reset_index()

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
    # Cast to float — prevents "numpy boolean subtract, the `-` operator,
    # is not supported" error when the column dtype is bool or object 0/1
    temp_series = df[col].dropna().astype(float)

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
                'Predictive Power': strength,
            })

    summary_df = pd.DataFrame(records)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            'IV', ascending=False).reset_index(drop=True)
    return summary_df
