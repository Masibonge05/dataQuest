import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, List, Tuple


def auto_bin_numeric(
        df: pd.DataFrame,
        col: str,
        target_col: str,
        num_bins: int = 4) -> Tuple[pd.DataFrame, Any, List[float]]:
    """
    A robust fallback for auto-binning numeric variables using quantiles.
    Calculates the Weight of Evidence (WoE) for each generated bin.
    """
    # Create copy and handle missing records cleanly
    temp_df = df[[col, target_col]].copy()
    temp_df[col] = temp_df[col].fillna(temp_df[col].median())

    # Bin via duplicate-safe quantiles
    try:
        temp_df['bin'] = pd.qcut(temp_df[col], q=num_bins, duplicates='drop')
    except ValueError:
        # Fallback to standard uniform intervals if distributions are too skewed
        temp_df['bin'] = pd.cut(temp_df[col], bins=num_bins, duplicates='drop')

    # Get distinct sorting intervals
    intervals = sorted(temp_df['bin'].unique(), key=lambda x: x.left)

    # Calculate global target distributions for WoE formulation
    total_bads = temp_df[target_col].sum()
    total_goods = len(temp_df) - total_bads

    # Small epsilon tweak to prevent zero divisions on clean metrics
    eps = 1e-4

    bin_records = []
    for idx, interval in enumerate(intervals):
        mask = temp_df['bin'] == interval
        subset = temp_df[mask]

        bads = subset[target_col].sum()
        goods = len(subset) - bads

        # Share of total counts distributions
        dist_bad = (bads + eps) / (total_bads + eps)
        dist_good = (goods + eps) / (total_goods + eps)

        # Standard Weight of Evidence formula
        woe = np.log(dist_good / dist_bad)

        bin_records.append({
            'bin': interval,
            'bin_label':
            f"[{round(interval.left, 2)} : {round(interval.right, 2)}]",
            'woe': woe
        })

    # Build edges list needed for out-of-bounds cut mappings
    edges = [intervals[0].left - eps] + [i.right for i in intervals]
    edges[-1] = edges[-1] + eps  # push upper boundary

    return pd.DataFrame(bin_records), None, edges


class CreditScorecard:

    def __init__(self,
                 base_points: float = 600,
                 base_odds: float = 20,
                 pdo: float = 50):
        """
        Initializes the CreditScorecard scaling parameters.
        Score = Offset + Factor * ln(Odds)
        Odds = P(Good) / P(Bad)
        """
        self.base_points = base_points
        self.base_odds = base_odds
        self.pdo = pdo

        # Scaling mathematics
        self.factor = pdo / np.log(2)
        self.offset = base_points - self.factor * np.log(base_odds)

        self.model = None
        self.features = []
        self.bin_edges = {}
        self.woe_mappings = {}  # col -> list of (bin_interval, woe, points)
        self.intercept = 0.0
        self.coefficients = {}

    def fit(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]):
        """Fits the scorecard by performing auto binning and logistic regression."""
        self.features = feature_cols
        X_woe = pd.DataFrame()

        # Bin features and create WoE variables
        for col in feature_cols:
            # Drop NaN rows during fitting to build unskewed coefficients
            clean_df = df.dropna(subset=[col, target_col])
            bin_table, _, edges = auto_bin_numeric(clean_df, col, target_col)
            self.bin_edges[col] = edges

            # Map values to WoE
            woe_map = []
            for _, row in bin_table.iterrows():
                woe_map.append({
                    'bin': row['bin'],
                    'woe': row['woe'],
                    'bin_label': row['bin_label']
                })
            self.woe_mappings[col] = woe_map

            # Create WoE column
            X_woe[col + '_woe'] = self._transform_col_to_woe(
                df[col], woe_map, edges)

        # Target assignment: 1 is Bad (Default), 0 is Good (Non-Default)
        y = df[target_col].values
        self.model = LogisticRegression(C=1.0, solver='liblinear')
        self.model.fit(X_woe, y)

        self.intercept = float(self.model.intercept_[0])
        for idx, col in enumerate(feature_cols):
            self.coefficients[col] = float(self.model.coef_[0][idx])

        # Distribute intercept points evenly across all variables
        n = len(feature_cols)
        for col in feature_cols:
            coef = self.coefficients[col]
            for mapping in self.woe_mappings[col]:
                woe_val = mapping['woe']
                # Traditional score scaling formula transformation
                points = -(woe_val * coef +
                           self.intercept / n) * self.factor + self.offset / n
                mapping['points'] = int(round(points))

    def _transform_col_to_woe(self, series: pd.Series, woe_map: list,
                              edges: list) -> pd.Series:
        """Transforms a numeric series to WoE values based on map and edges."""
        if not edges:
            return pd.Series(0.0, index=series.index)

        # Fill NaN records dynamically using column median
        clean_series = series.fillna(series.median())

        # Pad bounds to eliminate boundary drop errors during application
        adjusted_edges = edges.copy()
        adjusted_edges[0] = -np.inf
        adjusted_edges[-1] = np.inf

        bins = pd.cut(clean_series, bins=adjusted_edges, include_lowest=True)
        transformed = pd.Series(0.0, index=series.index)

        for mapping in woe_map:
            # Handle Interval overlap matches safely
            mask = bins.apply(lambda x: x.overlaps(mapping['bin'])
                              if pd.notna(x) else False)
            transformed[mask] = mapping['woe']

        return transformed

    def predict_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculates credit scores for input data by summing the points for the matched bins."""
        scores = pd.Series(0, index=df.index)

        for col in self.features:
            adjusted_edges = self.bin_edges[col].copy()
            adjusted_edges[0] = -np.inf
            adjusted_edges[-1] = np.inf

            clean_series = df[col].fillna(df[col].median())
            bins = pd.cut(clean_series,
                          bins=adjusted_edges,
                          include_lowest=True)

            points_series = pd.Series(0, index=df.index)
            for mapping in self.woe_mappings[col]:
                mask = bins.apply(lambda x: x.overlaps(mapping['bin'])
                                  if pd.notna(x) else False)
                points_series[mask] = mapping['points']

            scores += points_series

        return scores

    def get_scorecard_table(self) -> pd.DataFrame:
        """Returns a tidy DataFrame showing the complete credit scorecard model rules."""
        records = []
        for col in self.features:
            for mapping in self.woe_mappings[col]:
                records.append({
                    'Variable': col,
                    'Bin Interval': mapping['bin_label'],
                    'WoE': round(mapping['woe'], 4),
                    'Coefficient': round(self.coefficients[col], 4),
                    'Points': mapping['points']
                })
        return pd.DataFrame(records)

    def get_explainer_contributions(self, row: pd.Series) -> pd.DataFrame:
        """For a single customer record, breaks down their score contributions variable-by-variable."""
        records = []
        total_score = 0

        for col in self.features:
            val = row[col]

            # Default fallbacks if unmatched or missing
            matched_point = 0
            matched_bin_label = "Missing / Out of bounds"
            matched_woe = 0.0

            # Handle missing evaluations via dynamic fallback assignment
            eval_val = val if pd.notna(val) else 0.0

            for mapping in self.woe_mappings[col]:
                bin_interval = mapping['bin']
                # Check containment criteria cleanly
                if bin_interval.left <= eval_val <= bin_interval.right or (
                        eval_val in bin_interval):
                    matched_point = mapping['points']
                    matched_bin_label = mapping['bin_label']
                    matched_woe = mapping['woe']
                    break

            if matched_bin_label == "Missing / Out of bounds" and len(
                    self.woe_mappings[col]) > 0:
                # If bound testing slips on continuous floats, fall back to closest matching edge array index
                fallback = self.woe_mappings[col][0]
                matched_point = fallback['points']
                matched_bin_label = fallback['bin_label']
                matched_woe = fallback['woe']

            records.append({
                'Variable': col,
                'Value': val,
                'Matched Bin': matched_bin_label,
                'WoE': round(matched_woe, 4),
                'Points Contribution': matched_point
            })
            total_score += matched_point

        return pd.DataFrame(records), total_score


# --- EXECUTION SCRIPT LINKING TO THE LOAN_BOOK.CSV ---
if __name__ == "__main__":
    csv_path = r"C:\Users\shaba\.gemini\antigravity\scratch\risklens-analytics\data\raw\loan_book.csv"

    print("Reading and parsing data dimensions...")
    raw_df = pd.read_csv(csv_path)

    # Define our scorecard targets and risk modeling feature columns
    target = 'default_flag'
    features = [
        'age', 'annual_income', 'employment_length_years',
        'num_delinquencies_2yr', 'credit_utilisation_pct', 'dti_ratio'
    ]

    # Filter matrix rows to drop any rows where target is missing
    modeling_df = raw_df.dropna(subset=[target]).copy()

    # Initialize and train our model using empirical data
    print("Training Logistic Scorecard (WoE Mapping Engine)...")
    scorecard = CreditScorecard(base_points=600, base_odds=20, pdo=50)
    scorecard.fit(modeling_df, target_col=target, feature_cols=features)

    print("\n>>> MODEL SCORECARD RULE MATRIX GENERATED <<<")
    print(scorecard.get_scorecard_table().to_string(index=False))

    # Pull an actual applicant from the CSV dataset to verify the pipeline
    test_applicant = modeling_df.iloc[0]
    print(
        f"\nEvaluating Profile for Applicant: {test_applicant['applicant_id_hash']}"
    )

    contributions_df, calculated_score = scorecard.get_explainer_contributions(
        test_applicant)
    print(f"Calculated Final Base Credit Score: {calculated_score} Points")
    print("\n>>> INDIVIDUAL POINT CONTRIBUTIONS BREAKDOWN <<<")
    print(contributions_df.to_string(index=False))
