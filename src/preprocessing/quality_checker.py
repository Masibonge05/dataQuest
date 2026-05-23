import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates missing values and percentages for each column."""
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    report = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage': missing_percent
    })
    return report.sort_values(by='Missing Percentage', ascending=False)


def detect_outliers_iqr(
        series: pd.Series,
        factor: float = 1.5) -> Tuple[pd.Series, Dict[str, float]]:
    """Detects outliers in a numeric series using the IQR method."""
    # Cast to float — prevents pyarrow ArrowNotImplementedError when the
    # column dtype is large_string, bool, or any non-numeric arrow type
    if not pd.api.types.is_numeric_dtype(series):
        empty = pd.Series(False, index=series.index)
        return empty, {
            'q25': 0.0,
            'q75': 0.0,
            'iqr': 0.0,
            'lower_bound': 0.0,
            'upper_bound': 0.0,
            'outlier_count': 0,
            'outlier_percentage': 0.0
        }

    series = series.dropna().astype(float)

    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - (factor * iqr)
    upper_bound = q75 + (factor * iqr)

    outliers = (series < lower_bound) | (series > upper_bound)
    bounds = {
        'q25':
        float(q25),
        'q75':
        float(q75),
        'iqr':
        float(iqr),
        'lower_bound':
        float(lower_bound),
        'upper_bound':
        float(upper_bound),
        'outlier_count':
        int(outliers.sum()),
        'outlier_percentage':
        float((outliers.sum() / len(series)) * 100) if len(series) > 0 else 0.0
    }
    return outliers, bounds


def calculate_psi(expected: np.ndarray,
                  actual: np.ndarray,
                  num_bins: int = 10) -> Tuple[float, pd.DataFrame]:
    """
    Calculates the Population Stability Index (PSI) between two populations.

    PSI Values:
    - PSI < 0.1:  No significant change / stable
    - 0.1 <= PSI < 0.25: Moderate change / warning
    - PSI >= 0.25: Significant change / action required
    """
    # Cast to float and filter NaNs — prevents boolean/arrow type errors
    expected = expected.astype(float)
    actual = actual.astype(float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0, pd.DataFrame()

    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi_components = (actual_pct - expected_pct) * np.log(
        actual_pct / expected_pct)
    psi_value = float(np.sum(psi_components))

    bin_labels = [
        f"Bin {i+1} ({bin_edges[i]:.2f} to {bin_edges[i+1]:.2f})"
        for i in range(num_bins)
    ]
    details = pd.DataFrame({
        'Bin': bin_labels,
        'Expected Count': expected_counts,
        'Expected %': expected_pct * 100,
        'Actual Count': actual_counts,
        'Actual %': actual_pct * 100,
        'PSI Contribution': psi_components
    })

    return psi_value, details


def assess_data_health(df: pd.DataFrame,
                       numerical_cols: list) -> Dict[str, Any]:
    """Generates a summary of the data quality health of the dataset."""
    missing_report = check_missing_values(df)
    avg_missing = missing_report['Missing Percentage'].mean()

    outliers_summary = {}
    total_outliers = 0
    total_numeric_values = 0

    for col in numerical_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            _, bounds = detect_outliers_iqr(df[col])
            outliers_summary[col] = bounds
            total_outliers += bounds['outlier_count']
            total_numeric_values += len(df[col].dropna())

    outlier_rate = (total_outliers / total_numeric_values *
                    100) if total_numeric_values > 0 else 0

    health_score = max(0.0, 100.0 - (avg_missing * 1.5) - (outlier_rate * 2.0))

    return {
        'health_score': round(health_score, 1),
        'average_missing_percentage': round(avg_missing, 2),
        'overall_outlier_rate': round(outlier_rate, 2),
        'missing_report': missing_report,
        'outliers_details': outliers_summary
    }
