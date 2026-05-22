import streamlit as st
from app.utils.data_loader import generate_portfolio_data
from src.modeling.scorecard import CreditScorecard

def init_session_state():
    """Initializes global data and model objects in Streamlit session state."""
    # 1. Load Portfolio Data
    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = generate_portfolio_data()
        
    # 2. Define Features
    if 'features' not in st.session_state:
        st.session_state.features = [
            'fico_score', 'debt_to_income', 'loan_to_value', 
            'employment_length', 'revolving_utilization', 'delinquencies_2yrs'
        ]
        
    # 3. Fit Credit Scorecard Model
    if 'scorecard' not in st.session_state:
        with st.spinner("Calibrating credit risk scorecard models..."):
            scorecard = CreditScorecard(base_points=600, base_odds=20, pdo=50)
            scorecard.fit(
                st.session_state.portfolio_df, 
                target_col='default', 
                feature_cols=st.session_state.features
            )
            st.session_state.scorecard = scorecard
            
            # Predict scores for the portfolio
            scores = scorecard.predict_score(st.session_state.portfolio_df)
            st.session_state.portfolio_df['credit_score'] = scores
            
    # 4. Set default cutoff score if not present
    if 'cutoff_score' not in st.session_state:
        st.session_state.cutoff_score = 620
