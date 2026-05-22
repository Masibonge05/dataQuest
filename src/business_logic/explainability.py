import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression


# --- BACKWARD COMPATIBLE FALLBACK AUTO-BINNING ENGINE ---
def auto_bin_numeric(df: pd.DataFrame,
                     col: str,
                     target_col: str,
                     num_bins: int = 4):
    temp_df = df[[col, target_col]].copy().dropna()
    try:
        temp_df['bin'] = pd.qcut(temp_df[col], q=num_bins, duplicates='drop')
    except ValueError:
        temp_df['bin'] = pd.cut(temp_df[col], bins=num_bins, duplicates='drop')
    intervals = sorted(temp_df['bin'].unique(), key=lambda x: x.left)
    total_bads = temp_df[target_col].sum()
    total_goods = len(temp_df) - total_bads
    eps = 1e-4
    bin_records = []
    for interval in intervals:
        subset = temp_df[temp_df['bin'] == interval]
        bads = subset[target_col].sum()
        goods = len(subset) - bads
        dist_bad = (bads + eps) / (total_bads + eps)
        dist_good = (goods + eps) / (total_goods + eps)
        bin_records.append({
            'bin': interval,
            'bin_label':
            f"[{round(interval.left,2)}:{round(interval.right,2)}]",
            'woe': np.log(dist_good / dist_bad)
        })
    edges = [intervals[0].left - eps] + [i.right for i in intervals]
    edges[-1] = edges[-1] + eps
    return pd.DataFrame(bin_records), None, edges


# --- EMPIRICAL CREDIT SCORECARD CLASS ---
class CreditScorecard:

    def __init__(self,
                 base_points: float = 600,
                 base_odds: float = 20,
                 pdo: float = 50):
        self.factor = pdo / np.log(2)
        self.offset = base_points - self.factor * np.log(base_odds)
        self.features = []
        self.bin_edges = {}
        self.woe_mappings = {}
        self.coefficients = {}
        self.intercept = 0.0

    def fit(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]):
        self.features = feature_cols
        X_woe = pd.DataFrame()
        for col in feature_cols:
            clean_df = df.dropna(subset=[col, target_col])
            bin_table, _, edges = auto_bin_numeric(clean_df, col, target_col)
            self.bin_edges[col] = edges
            self.woe_mappings[col] = bin_table.to_dict('records')
            X_woe[col + '_woe'] = self._transform_col_to_woe(
                df[col], self.woe_mappings[col], edges)

        model = LogisticRegression(C=1.0, solver='liblinear')
        model.fit(X_woe, df[target_col].values)
        self.intercept = float(model.intercept_[0])
        for idx, col in enumerate(feature_cols):
            self.coefficients[col] = float(model.coef_[0][idx])

        n = len(feature_cols)
        for col in feature_cols:
            for mapping in self.woe_mappings[col]:
                points = -(mapping['woe'] * self.coefficients[col] +
                           self.intercept / n) * self.factor + self.offset / n
                mapping['points'] = int(round(points))

    def _transform_col_to_woe(self, series: pd.Series, woe_map: list,
                              edges: list) -> pd.Series:
        clean = series.fillna(series.median())
        adj_edges = edges.copy()
        adj_edges[0], adj_edges[-1] = -np.inf, np.inf
        bins = pd.cut(clean, bins=adj_edges, include_lowest=True)
        transformed = pd.Series(0.0, index=series.index)
        for mapping in woe_map:
            mask = bins.apply(lambda x: x.overlaps(mapping['bin'])
                              if pd.notna(x) else False)
            transformed[mask] = mapping['woe']
        return transformed

    def get_scorecard_table(self) -> pd.DataFrame:
        records = []
        for col in self.features:
            for mapping in self.woe_mappings[col]:
                records.append({
                    'Variable': col,
                    'Bin Interval': mapping['bin_label'],
                    'Points': mapping['points']
                })
        return pd.DataFrame(records)

    def get_explainer_contributions(self, row: pd.Series) -> pd.DataFrame:
        records = []
        for col in self.features:
            val = row[col]
            matched_point = 0
            eval_val = val if pd.notna(
                val) else row.to_frame().T[col].median()  # quick vector fill

            for mapping in self.woe_mappings[col]:
                if mapping['bin'].left <= eval_val <= mapping['bin'].right or (
                        eval_val in mapping['bin']):
                    matched_point = mapping['points']
                    break
            if matched_point == 0 and len(self.woe_mappings[col]) > 0:
                matched_point = self.woe_mappings[col][0]['points']

            records.append({
                'Variable': col,
                'Points Contribution': matched_point
            })
        return pd.DataFrame(records)


# --- UPDATED COMPLIANCE & WATERFALL FUNCTIONS ---
def get_regulatory_reasons(scorecard_df: pd.DataFrame,
                           contributions_df: pd.DataFrame,
                           num_reasons: int = 4) -> pd.DataFrame:
    """
    Identifies the top adverse reasons for a credit score (why the score isn't higher).
    This mimics US ECOA Regulation B Adverse Action notices.
    """
    # Group scorecard rules by variable and find max points available in empirical training
    max_points = scorecard_df.groupby('Variable')['Points'].max().reset_index()
    max_points.rename(columns={'Points': 'Max Possible Points'}, inplace=True)

    # Merge with applicant contributions
    reasons_df = pd.merge(contributions_df, max_points, on='Variable')

    # Impact = Max Possible Points - Points Contribution
    reasons_df['Points Lost'] = reasons_df['Max Possible Points'] - reasons_df[
        'Points Contribution']

    # Sort by points lost (descending) - variables where they lost the most potential points
    reasons_df = reasons_df.sort_values(by='Points Lost', ascending=False)

    # Corrected Reason Map strictly referencing raw CSV headers
    reason_map = {
        'age': 'Short age of credit file or younger age profile',
        'annual_income': 'Low annual income constraints',
        'employment_length_years': 'Short duration of current employment',
        'num_delinquencies_2yr': 'History of recent credit delinquencies',
        'credit_utilisation_pct': 'High utilization of revolving credit lines',
        'dti_ratio': 'High debt relative to monthly income',
        'num_open_accounts': 'Too many or too few open credit lines'
    }

    reasons_df['Adverse Reason'] = reasons_df['Variable'].map(
        reason_map).fillna(reasons_df['Variable'].apply(
            lambda x: f"Sub-optimal {x.replace('_', ' ').title()}"))

    return reasons_df.head(num_reasons)


def generate_waterfall_data(contributions_df: pd.DataFrame,
                            base_points: float) -> pd.DataFrame:
    """
    Generates structured data for a waterfall chart explaining a credit score.
    Starting from base points, each feature adds points, ending at the total score.
    """
    records = []

    # Start node: System Baseline (Offset score constant)
    records.append({
        'Label': 'System Baseline',
        'Value': float(base_points),
        'Type': 'absolute'
    })

    # Map raw points as incremental steps
    for _, row in contributions_df.iterrows():
        records.append({
            'Label': str(row['Variable']).replace('_', ' ').title(),
            'Value': float(row['Points Contribution']),
            'Type': 'relative'
        })

    return pd.DataFrame(records)


# --- RUN PIPELINE EXECUTIVE CONTROL ---
if __name__ == "__main__":
    csv_path = r"C:\Users\shaba\.gemini\antigravity\scratch\risklens-analytics\data\raw\loan_book.csv"

    print("Extracting credit records matrices...")
    loan_book = pd.read_csv(csv_path)

    target = 'default_flag'
    features = [
        'age', 'annual_income', 'employment_length_years',
        'num_delinquencies_2yr', 'credit_utilisation_pct', 'dti_ratio'
    ]

    # Prepare clean dataset
    clean_df = loan_book.dropna(subset=[target]).copy()

    print("Training parent scorecard model...")
    sc = CreditScorecard()
    sc.fit(clean_df, target_col=target, feature_cols=features)

    # Extract underlying rule parameter distributions
    trained_scorecard_rules = sc.get_scorecard_table()

    # Pull sample row entry
    sample_borrower = clean_df.iloc[0]
    print(
        f"\nEvaluating Profiles for Applicant ID Key: {sample_borrower['applicant_id_hash']}"
    )

    # Compute points contributions
    borrower_contributions = sc.get_explainer_contributions(sample_borrower)

    # 1. Generate Adverse Actions Table
    adverse_action_notice = get_regulatory_reasons(
        scorecard_df=trained_scorecard_rules,
        contributions_df=borrower_contributions,
        num_reasons=4)
    print("\n--- REGULATORY ECOA DISCLOSURE DATA ---")
    print(adverse_action_notice[['Variable', 'Points Lost',
                                 'Adverse Reason']].to_string(index=False))

    # 2. Generate Plotting Waterfall Table
    # The intercept and scaling factor establish the model baseline mathematical floor
    waterfall_table = generate_waterfall_data(
        contributions_df=borrower_contributions,
        base_points=round(sc.offset, 2))
    print("\n--- WATERFALL PLOT TABLE DATA ---")
    print(waterfall_table.to_string(index=False))
