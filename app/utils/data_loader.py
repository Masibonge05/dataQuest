import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def generate_portfolio_data(n_samples: int = 2500, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic portfolio of retail credit applications.
    Computes a synthetic default probability (PD) based on real-world credit risk factors
    and assigns a binary default label using a Bernoulli trial.
    """
    np.random.seed(seed)
    
    # 1. Generate demographic & financial features
    age = np.random.normal(42, 12, n_samples).clip(18, 80)
    fico_score = np.random.normal(670, 75, n_samples).clip(300, 850)
    
    # Debt to Income (DTI) - usually right-skewed
    debt_to_income = np.random.exponential(0.3, n_samples).clip(0.05, 0.85)
    
    # Loan to Value (LTV)
    loan_to_value = np.random.normal(0.75, 0.2, n_samples).clip(0.1, 1.4)
    
    # Employment length (years)
    employment_length = np.random.exponential(6.0, n_samples).clip(0, 40)
    
    # Revolving Utilization
    revolving_utilization = np.random.normal(0.4, 0.28, n_samples).clip(0.0, 1.2)
    
    # Delinquencies in the past 2 years (poisson distribution)
    delinquencies_2yrs = np.random.poisson(0.3, n_samples).clip(0, 5)
    
    df = pd.DataFrame({
        'application_id': [f"APP-{100000 + i}" for i in range(n_samples)],
        'age': np.round(age).astype(int),
        'fico_score': np.round(fico_score).astype(int),
        'debt_to_income': np.round(debt_to_income, 3),
        'loan_to_value': np.round(loan_to_value, 3),
        'employment_length': np.round(employment_length, 1),
        'revolving_utilization': np.round(revolving_utilization, 3),
        'delinquencies_2yrs': delinquencies_2yrs
    })
    
    # 2. Compute synthetic default probability (PD) using a logistic function
    # coefficients are chosen to represent realistic credit risk directions:
    # Lower FICO = Higher Default
    # Higher DTI = Higher Default
    # Higher LTV = Higher Default
    # Lower Employment = Higher Default
    # Higher Utilization = Higher Default
    # Higher Delinquencies = Higher Default
    
    # Normalize features for logodds computation
    x_fico = (df['fico_score'] - 650) / 75
    x_dti = (df['debt_to_income'] - 0.3) / 0.2
    x_ltv = (df['loan_to_value'] - 0.75) / 0.2
    x_emp = (df['employment_length'] - 5.0) / 5.0
    x_util = (df['revolving_utilization'] - 0.4) / 0.25
    x_delinq = df['delinquencies_2yrs']
    
    # Log-odds of default
    log_odds = (
        -1.8  # baseline default rate
        - 1.5 * x_fico 
        + 0.8 * x_dti 
        + 0.9 * x_ltv 
        - 0.5 * x_emp 
        + 0.7 * x_util 
        + 0.6 * x_delinq
    )
    
    # Convert log-odds to probability
    default_prob = 1 / (1 + np.exp(-log_odds))
    
    # Assign binary default status (Bernoulli trial)
    df['default'] = np.random.binomial(1, default_prob)
    
    # Let's also add an application date within the last 12 months for trends
    dates = pd.date_range(start='2025-05-01', end='2026-05-01', periods=n_samples)
    df['application_date'] = np.random.choice(dates, n_samples)
    df = df.sort_values(by='application_date').reset_index(drop=True)
    
    # Add loan size
    loan_size = np.random.normal(25000, 10000, n_samples).clip(5000, 75000)
    df['loan_size'] = np.round(loan_size, -2).astype(int)
    
    return df
