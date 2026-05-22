import streamlit as st
from app.components.header import render_header

def render_page():
    render_header(
        page_title="Research & Methodology", 
        page_subtitle="Mathematical Frameworks, Scorecard Scaling, and Regulatory Standards"
    )
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("1. Weight of Evidence (WoE) & Information Value (IV)")
    st.markdown("""
    The **Weight of Evidence (WoE)** measures the predictive power of an independent variable in relation to a binary dependent variable (e.g. Default vs. Non-default). 
    The mathematical formulation is:
    
    $$WoE_i = \\ln\\left( \\frac{\\text{Good Distribution}_i}{\\text{Bad Distribution}_i} \\right) = \\ln\\left( \\frac{G_i / G_{\\text{total}}}{B_i / B_{\\text{total}}} \\right)$$
    
    Where:
    *   $G_i$ is the number of good customers (non-defaulters) in group $i$.
    *   $B_i$ is the number of bad customers (defaulters) in group $i$.
    
    **Information Value (IV)** is used to rank variables based on their predictive power:
    
    $$IV = \\sum_{i=1}^{n} (\\text{Good Distribution}_i - \\text{Bad Distribution}_i) \\times WoE_i$$
    
    Variables are generally grouped as follows:
    *   **< 0.02**: Useless for prediction
    *   **0.02 to 0.1**: Weak predictive power
    *   **0.1 to 0.3**: Medium predictive power
    *   **0.3 to 0.5**: Strong predictive power
    *   **> 0.5**: Suspiciously high (may indicate data leakage)
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("2. Scorecard Scaling Mathematics")
    st.markdown("""
    Credit scorecards do not display log-odds directly because they are difficult for front-line lending staff and regulators to interpret. Instead, scores are scaled linearly.
    
    The scaling formulation is:
    
    $$\\text{Score} = \\text{Offset} + \\text{Factor} \\times \\ln(\\text{Odds})$$
    
    Where $\\text{Odds} = \\frac{P(\\text{Good})}{P(\\text{Bad})}$. The scaling parameters are solved using two user-defined rules:
    1.  A **Base Score** at specific **Base Odds** (e.g., a score of 600 at odds of 20:1).
    2.  A **Points to Double Odds (PDO)** (e.g., 50 points to double the odds).
    
    $$\\text{Factor} = \\frac{\\text{PDO}}{\\ln(2)}$$
    $$\\text{Offset} = \\text{Base Score} - \\text{Factor} \\times \\ln(\\text{Base Odds})$$
    
    Once the parameters are set, individual scorecard points for variable $j$, bin $i$ are computed from the Logistic Regression coefficients ($\\beta_j$) and intercept ($\\alpha$):
    
    $$\\text{Points}_{j, i} = -\\left( WoE_{j, i} \\times \\beta_j + \\frac{\\alpha}{n} \\right) \\times \\text{Factor} + \\frac{\\text{Offset}}{n}$$
    
    Where $n$ represents the total number of features in the model. Summing these points across all variables yields the final credit score.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("3. Regulatory Compliance: Basel Accord Frameworks")
    st.markdown("""
    Under the **Basel III / Basel IV** internal ratings-based (IRB) approach, institutions calculate credit risk capital requirements using three primary pillars:
    
    1.  **Probability of Default (PD):** The likelihood that a borrower will default over a given time horizon (typically 12 months). Measured via scorecard models.
    2.  **Loss Given Default (LGD):** The share of exposure that is lost when a borrower defaults, after taking into account collateral recovery costs.
    3.  **Exposure at Default (EAD):** The total gross exposure of the facility at the time of default.
    
    **Expected Loss (EL)** is defined as:
    
    $$\\text{EL} = \\text{PD} \\times \\text{LGD} \\times \\text{EAD}$$
    
    The platform models default risks using these frameworks to ensure transparency, consistency, and compliance with institutional standards.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
