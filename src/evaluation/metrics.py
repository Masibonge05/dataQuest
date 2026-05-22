import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple, List


# --- COPIED LIGHTWEIGHT INTEGRATION OF SCORECARD FROM PREVIOUS STEP ---
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

    def predict_score(self, df: pd.DataFrame) -> pd.Series:
        scores = pd.Series(0, index=df.index)
        for col in self.features:
            adj_edges = self.bin_edges[col].copy()
            adj_edges[0], adj_edges[-1] = -np.inf, np.inf
            bins = pd.cut(df[col].fillna(df[col].median()),
                          bins=adj_edges,
                          include_lowest=True)
            points_series = pd.Series(0, index=df.index)
            for mapping in self.woe_mappings[col]:
                mask = bins.apply(lambda x: x.overlaps(mapping['bin'])
                                  if pd.notna(x) else False)
                points_series[mask] = mapping['points']
            scores += points_series
        return scores


# --- PRODUCTION METRICS METRIC INTERFACE FUNCTIONS ---
def calculate_model_performance(y_true: np.ndarray,
                                y_score: np.ndarray) -> Dict[str, Any]:
    """
    Computes key validation metrics for risk models using empirical scores.
    Correctly accounts for higher scores indicating higher creditworthiness (lower default risk).
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(y_true) == 0:
        return {}

    # Standard transformation: Invert actual scores to transform them into default-aligned probabilities
    # required by sci-kit metrics engines (lower points map down to high default probabilities)
    y_prob_default = -y_score

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob_default)
    roc_auc = auc(fpr, tpr)
    gini = 2.0 * roc_auc - 1.0

    # KS Statistic Computation
    df = pd.DataFrame({'y_true': y_true, 'score': y_score})
    df = df.sort_values(by='score').reset_index(
        drop=True)  # Sort lowest (riskiest) to highest

    total_bads = df['y_true'].sum()
    total_goods = len(df) - total_bads

    df['cum_bads'] = df['y_true'].cumsum()
    df['cum_goods'] = (1 - df['y_true']).cumsum()

    df['cum_bads_rate'] = df['cum_bads'] / total_bads if total_bads > 0 else 0
    df['cum_goods_rate'] = df[
        'cum_goods'] / total_goods if total_goods > 0 else 0

    df['ks_diff'] = np.abs(df['cum_bads_rate'] - df['cum_goods_rate'])
    ks_stat = float(df['ks_diff'].max())

    # Pick target coordinate thresholds maximizing distance calculations
    max_ks_idx = df['ks_diff'].idxmax()
    ks_score_cutoff = float(df.loc[max_ks_idx, 'score'])

    # 100-point uniform coordinate grid for front-end charts plotting
    sample_indices = np.linspace(0, len(df) - 1, 100, dtype=int)
    ks_curve_df = df.loc[
        sample_indices,
        ['score', 'cum_bads_rate', 'cum_goods_rate', 'ks_diff']]

    # CAP (Cumulative Accuracy Profile) Curve Calculations
    cap_df = df.sort_values(by='score', ascending=True).reset_index(drop=True)
    cap_df['cum_bads_pct'] = cap_df['y_true'].cumsum(
    ) / total_bads if total_bads > 0 else 0
    cap_df['pop_pct'] = (cap_df.index + 1) / len(cap_df)

    perfect_cap = np.zeros(len(cap_df))
    bads_count = int(total_bads)
    perfect_cap[:bads_count] = np.arange(1, bads_count + 1) / total_bads
    perfect_cap[bads_count:] = 1.0

    cap_curve_data = pd.DataFrame({
        'pop_pct':
        cap_df.loc[sample_indices, 'pop_pct'] * 100,
        'model_cap':
        cap_df.loc[sample_indices, 'cum_bads_pct'] * 100,
        'perfect_cap':
        perfect_cap[sample_indices] * 100,
        'random_cap':
        cap_df.loc[sample_indices, 'pop_pct'] * 100
    })

    precision, recall, pr_thresholds = precision_recall_curve(
        y_true, y_prob_default)

    return {
        'auc': round(roc_auc, 4),
        'gini': round(gini, 4),
        'ks_statistic': round(ks_stat, 4),
        'ks_score_cutoff': round(ks_score_cutoff, 1),
        'roc_curve': pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr
        }),
        'ks_curve': ks_curve_df,
        'cap_curve': cap_curve_data,
        'pr_curve': pd.DataFrame({
            'precision': precision,
            'recall': recall
        })
    }


def get_confusion_matrix_details(
        y_true: np.ndarray, y_score: np.ndarray,
        cutoff_score: float) -> Tuple[np.ndarray, Dict[str, float]]:
    """Calculates confusion matrix details for credit risks based on the point cut-offs."""
    # Scores below the cutoff threshold indicate a predicted default (1)
    y_pred = np.where(y_score < cutoff_score, 1, 0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision +
                                                             recall) > 0 else 0

    fpr = fp / (fp + tn) if (
        fp + tn) > 0 else 0  # Type I Error: False Approval Rate
    fnr = fn / (fn + tp) if (
        fn + tp) > 0 else 0  # Type II Error: False Rejection Rate

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr)
    }

    return cm, metrics


# --- SYSTEM PIPELINE ENTRYPOINT ---
if __name__ == "__main__":
    csv_path = r"C:\Users\shaba\.gemini\antigravity\scratch\risklens-analytics\data\raw\loan_book.csv"

    print("Extracting credit records matrices...")
    full_df = pd.read_csv(csv_path)

    # Establish targets and numeric modeling columns
    target_col = 'default_flag'
    features = [
        'age', 'annual_income', 'employment_length_years',
        'num_delinquencies_2yr', 'credit_utilisation_pct', 'dti_ratio'
    ]

    # Process modeling data frame copies safely
    clean_df = full_df.dropna(subset=[target_col]).copy()

    # Train/Test Validation Split (80/20 standard profile distribution)
    train_df, test_df = train_test_split(clean_df,
                                         test_size=0.20,
                                         random_state=42,
                                         stratify=clean_df[target_col])
    print(
        f"Data split executed: {len(train_df)} training samples, {len(test_df)} testing validation samples."
    )

    # Execute Model Fit Step
    print("Fitting core scorecard coefficients...")
    scorecard = CreditScorecard()
    scorecard.fit(train_df, target_col=target_col, feature_cols=features)

    # Generate scores predictions out across unseen test portfolios
    print("Scoring out performance validation data array profiles...")
    test_scores = scorecard.predict_score(test_df).values
    test_targets = test_df[target_col].values

    # 1. Run Complete Validation Performance Metrics Calculation
    print("\nProcessing Risk Performance Evaluation Metrics...")
    results = calculate_model_performance(test_targets, test_scores)

    print("\n==========================================")
    print("      RISK MODEL VALIDATION RESULTS       ")
    print("==========================================")
    print(f"Receiver Operating Area Under Curve (AUC): {results['auc']}")
    print(f"Gini Separation Coefficient            : {results['gini']}")
    print(
        f"Kolmogorov-Smirnov (KS) Statistic      : {results['ks_statistic']}")
    print(
        f"Optimal KS Credit Score Cut-Off Threshold: {results['ks_score_cutoff']} Points"
    )

    # 2. Run Confusion Matrices Analysis on the Optimal Score Cutoff
    optimal_cutoff = results['ks_score_cutoff']
    cm, cm_metrics = get_confusion_matrix_details(test_targets,
                                                  test_scores,
                                                  cutoff_score=optimal_cutoff)

    print("\n==========================================")
    print(f" CONFUSION MATRIX DETAILS (@ CUTOFF {optimal_cutoff})")
    print("==========================================")
    print(f"Confusion Matrix Array:\n{cm}")
    print(
        f"Model Overall Accuracy    : {round(cm_metrics['accuracy'] * 100, 2)}%"
    )
    print(
        f"Precision (Positive Pay)  : {round(cm_metrics['precision'] * 100, 2)}%"
    )
    print(
        f"Recall (Sensitivity rate) : {round(cm_metrics['recall'] * 100, 2)}%")
    print(
        f"False Acceptance Rate(FPR): {round(cm_metrics['false_positive_rate'] * 100, 2)}% (Approved but Defaulted)"
    )
    print(
        f"False Rejection Rate (FNR): {round(cm_metrics['false_negative_rate'] * 100, 2)}% (Rejected but Good)"
    )
