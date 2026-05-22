import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import uuid

from app.components.header import render_header

# ==========================================================
# CONFIGURATION
# ==========================================================
CSV_PATH = "data/raw/loan_book.csv"

PRIMARY_TURQUOISE = "#00A7B5"
ACCENT_ORANGE = "#F58220"
BLACK = "#111111"
CARD_BG = "#FFFFFF"


# ==========================================================
# PAGE
# ==========================================================
def render_page():

    # ======================================================
    # STYLING
    # ======================================================
    st.markdown(f"""
        <style>

        .main .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1600px;
        }}

        h1, h2, h3 {{
            color: {BLACK} !important;
            font-weight: 700 !important;
        }}

        div[data-testid="stForm"] {{
            background: white;
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0px 2px 12px rgba(0,0,0,0.04);
        }}

        .stNumberInput label,
        .stSelectbox label,
        .stSlider label,
        .stDateInput label,
        .stFileUploader label {{
            font-weight: 600 !important;
            color: {BLACK} !important;
        }}

        .stButton > button,
        .stFormSubmitButton > button {{

            background: linear-gradient(
                135deg,
                {PRIMARY_TURQUOISE},
                #007A87
            ) !important;

            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            height: 50px !important;
            font-weight: 700 !important;
            font-size: 15px !important;
        }}

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {{

            transform: translateY(-1px);
            box-shadow: 0px 6px 18px rgba(0,0,0,0.12);
        }}

        div[data-testid="stSuccess"] {{
            border-radius: 14px;
            border-left: 5px solid {PRIMARY_TURQUOISE};
        }}

        div[data-testid="stInfo"] {{
            border-radius: 14px;
            border-left: 5px solid {ACCENT_ORANGE};
        }}

        </style>
        """,
                unsafe_allow_html=True)

    # ======================================================
    # HEADER
    # ======================================================
    render_header(page_title="Client Onboarding Center",
                  page_subtitle=("Enterprise Borrower Registration & "
                                 "Credit Application Intake"))

    # ======================================================
    # HERO SECTION
    # ======================================================
    st.info("""
        ### Enterprise Client Registration

        Register new borrowers directly into the lending portfolio
        dataset for credit scoring, policy simulation,
        explainability analysis, and enterprise portfolio monitoring.
        """)

    # ======================================================
    # LOAD EXISTING CSV
    # ======================================================
    try:

        existing_df = pd.read_csv(CSV_PATH)

    except Exception as e:

        st.error(f"Failed to load loan book CSV: {e}")

        return

    # ======================================================
    # BULK CSV UPLOAD
    # ======================================================
    st.markdown("## Bulk Client CSV Upload")

    uploaded_csv = st.file_uploader("Upload Additional Client CSV",
                                    type=["csv"])

    if uploaded_csv is not None:

        try:

            uploaded_df = pd.read_csv(uploaded_csv)

            st.success(f"CSV uploaded successfully. "
                       f"Detected {len(uploaded_df):,} records.")

            st.dataframe(uploaded_df.head())

            if st.button("Append Uploaded CSV To Loan Book",
                         use_container_width=True):

                # ==========================================
                # ADD MISSING COLUMNS
                # ==========================================
                for col in existing_df.columns:

                    if col not in uploaded_df.columns:

                        uploaded_df[col] = np.nan

                # ==========================================
                # KEEP ONLY SYSTEM COLUMNS
                # ==========================================
                uploaded_df = uploaded_df[existing_df.columns]

                # ==========================================
                # APPEND
                # ==========================================
                combined_df = pd.concat([existing_df, uploaded_df],
                                        ignore_index=True)

                # ==========================================
                # SAVE
                # ==========================================
                combined_df.to_csv(CSV_PATH, index=False)

                st.success(f"{len(uploaded_df):,} records successfully "
                           f"added to the enterprise loan book.")

        except Exception as e:

            st.error(f"CSV upload failed: {e}")

    st.divider()

    # ======================================================
    # FORM
    # ======================================================
    st.markdown("## New Client Application")

    with st.form("client_onboarding_form"):

        st.markdown("### Personal & Financial Information")

        c1, c2, c3 = st.columns(3)

        # ==================================================
        # COLUMN 1
        # ==================================================
        with c1:

            age = st.number_input("Age",
                                  min_value=18.0,
                                  max_value=100.0,
                                  value=30.0,
                                  step=0.1)

            annual_income = st.number_input("Annual Income (R)",
                                            min_value=0.0,
                                            value=50000.0,
                                            step=1000.0)

            employment_length_years = st.number_input(
                "Employment Length (Years)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.1)

            home_ownership = st.selectbox("Home Ownership",
                                          ["MORTGAGE", "RENT", "OWN", "OTHER"])

            region = st.selectbox("Region", [
                "North-Urban", "South-Urban", "East-Urban", "West-Urban",
                "Central-Urban", "North-Suburban", "South-Suburban",
                "East-Suburban"
            ])

        # ==================================================
        # COLUMN 2
        # ==================================================
        with c2:

            num_open_accounts = st.number_input("Number of Open Accounts",
                                                min_value=0.0,
                                                value=5.0,
                                                step=1.0)

            num_delinquencies_2yr = st.number_input("Delinquencies (2 Years)",
                                                    min_value=0,
                                                    value=0,
                                                    step=1)

            total_revolving_balance = st.number_input(
                "Total Revolving Balance",
                min_value=0.0,
                value=5000.0,
                step=100.0)

            credit_utilisation_pct = st.slider("Credit Utilisation %",
                                               min_value=0.0,
                                               max_value=100.0,
                                               value=30.0,
                                               step=0.1)

            months_since_oldest_account = st.number_input(
                "Months Since Oldest Account",
                min_value=0.0,
                value=120.0,
                step=1.0)

        # ==================================================
        # COLUMN 3
        # ==================================================
        with c3:

            loan_amount = st.number_input("Loan Amount",
                                          min_value=0.0,
                                          value=15000.0,
                                          step=500.0)

            interest_rate = st.number_input("Interest Rate (%)",
                                            min_value=0.0,
                                            max_value=100.0,
                                            value=12.5,
                                            step=0.01)

            loan_purpose = st.selectbox("Loan Purpose", [
                "debt_consolidation", "major_purchase", "home_improvement",
                "medical", "education", "vacation", "small_business", "other"
            ])

            dti_ratio = st.slider("Debt-To-Income Ratio",
                                  min_value=0.0,
                                  max_value=1.0,
                                  value=0.25,
                                  step=0.001)

            pct_accounts_current = st.slider("Percent Accounts Current",
                                             min_value=0.0,
                                             max_value=100.0,
                                             value=90.0,
                                             step=0.1)

        st.markdown("---")

        st.markdown("### Verification & Additional Information")

        d1, d2, d3 = st.columns(3)

        with d1:

            months_since_last_delinquency = st.number_input(
                "Months Since Last Delinquency",
                min_value=0.0,
                value=12.0,
                step=1.0)

            branch_code_id = st.number_input("Branch Code ID",
                                             min_value=100,
                                             max_value=999,
                                             value=300,
                                             step=1)

        with d2:

            months_at_current_address = st.number_input(
                "Months At Current Address", min_value=0, value=36, step=1)

            email_domain_type = st.selectbox("Email Domain Type",
                                             ["corporate", "free", "other"])

        with d3:

            phone_verified = st.selectbox("Phone Verified", [True, False])

            application_date = st.date_input("Application Date",
                                             value=datetime.now())

        # ==================================================
        # SUBMIT BUTTON
        # ==================================================
        submitted = st.form_submit_button("Register Client Application",
                                          use_container_width=True)

    # ======================================================
    # SAVE CLIENT
    # ======================================================
    if submitted:

        try:

            new_row = {
                "applicant_id_hash":
                uuid.uuid4().hex[:16],
                "age":
                float(age),
                "annual_income":
                float(annual_income),
                "employment_length_years":
                float(employment_length_years),
                "num_open_accounts":
                float(num_open_accounts),
                "num_delinquencies_2yr":
                int(num_delinquencies_2yr),
                "total_revolving_balance":
                float(total_revolving_balance),
                "credit_utilisation_pct":
                float(credit_utilisation_pct),
                "months_since_oldest_account":
                float(months_since_oldest_account),
                "num_hard_inquiries_6mo":
                int(np.random.randint(0, 5)),
                "loan_amount":
                float(loan_amount),
                "interest_rate":
                float(interest_rate),
                "dti_ratio":
                float(dti_ratio),
                "months_since_last_delinquency":
                float(months_since_last_delinquency),
                "pct_accounts_current":
                float(pct_accounts_current),
                "branch_code_id":
                int(branch_code_id),
                "months_at_current_address":
                int(months_at_current_address),
                "home_ownership":
                str(home_ownership).upper(),
                "region":
                str(region),
                "loan_purpose":
                str(loan_purpose).lower(),
                "email_domain_type":
                str(email_domain_type).lower(),
                "application_date":
                application_date.strftime("%m/%d/%Y"),
                "application_dow":
                application_date.strftime("%A"),
                "phone_verified":
                bool(phone_verified),
                "default_flag":
                0,
                "set":
                "new"
            }

            # ==================================================
            # CREATE DF
            # ==================================================
            new_df = pd.DataFrame([new_row])

            # ==================================================
            # MATCH COLUMNS
            # ==================================================
            for col in existing_df.columns:

                if col not in new_df.columns:

                    new_df[col] = np.nan

            new_df = new_df[existing_df.columns]

            # ==================================================
            # APPEND
            # ==================================================
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)

            # ==================================================
            # SAVE
            # ==================================================
            updated_df.to_csv(CSV_PATH, index=False)

            st.success("Client application successfully added "
                       "to enterprise loan book.")

            # ==================================================
            # SUMMARY
            # ==================================================
            st.markdown("## Application Summary")

            s1, s2, s3, s4 = st.columns(4)

            with s1:
                st.metric("Loan Amount", f"R{loan_amount:,.0f}")

            with s2:
                st.metric("Interest Rate", f"{interest_rate:.2f}%")

            with s3:
                st.metric("Annual Income", f"R{annual_income:,.0f}")

            with s4:
                st.metric("DTI Ratio", f"{dti_ratio:.3f}")

        except Exception as e:

            st.error(f"Error saving client application: {e}")
