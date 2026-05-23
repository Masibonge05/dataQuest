import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit_option_menu import option_menu

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="RiskLens Analytics",
                   page_icon="🏦",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ==========================================================
# PROJECT PATHS
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR / "app"
STYLES_DIR = APP_DIR / "styles"
ASSETS_DIR = APP_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "fnb_logo.png"
CSV_PATH = BASE_DIR / "data" / "raw" / "loan_book.csv"

# ==========================================================
# LOAD CSS FILES
# ==========================================================
theme_css = STYLES_DIR / "theme.css"
styles_css = STYLES_DIR / "styles.css"

if theme_css.exists():
    with open(theme_css, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.error(f"Missing theme.css: {theme_css}")

if styles_css.exists():
    with open(styles_css, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning(f"Missing styles.css: {styles_css}")

# ==========================================================
# GLOBAL ENTERPRISE UI ENHANCEMENTS
# ==========================================================
st.markdown("""
<style>

/* =========================================================
   SIDEBAR BASE (RESTORED ORIGINAL DARK THEME)
========================================================= */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #111827 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* =========================================================
   BRANDING
========================================================= */
.brand-wrapper  {
    padding-top: 10px;
    padding-bottom: 20px;
    text-align: center;
}

.brand-title {
    font-size: 28px;
    font-weight: 800;
    color: white;
    letter-spacing: 1px;
    margin-top: 10px;
}

.brand-subtitle {
    font-size: 13px;
    color: #CBD5E1;
    margin-top: 2px;
    letter-spacing: 0.5px;
}

/* =========================================================
   MAIN CONTENT
========================================================= */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

h1, h2, h3 {
    color: #111111 !important;
    font-weight: 700 !important;
}

/* =========================================================
   METRICS
========================================================= */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid rgba(0,0,0,0.08);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.04);
}

[data-testid="metric-container"] label {
    color: #5B657A !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #111111 !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* =========================================================
   FORMS
========================================================= */
div[data-testid="stForm"] {
    background: white;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0px 2px 12px rgba(0,0,0,0.04);
}

/* =========================================================
   INPUT LABELS
========================================================= */
.stNumberInput label,
.stSelectbox label,
.stSlider label {
    font-weight: 600 !important;
    color: #111111 !important;
}

/* =========================================================
   BUTTONS
========================================================= */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #00A7B5, #007A87) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    height: 50px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0px 6px 18px rgba(0,0,0,0.12);
}

/* =========================================================
   ALERTS
========================================================= */
div[data-testid="stSuccess"] {
    border-radius: 14px;
    border-left: 5px solid #00A7B5;
}

div[data-testid="stInfo"] {
    border-radius: 14px;
    border-left: 5px solid #F58220;
}

/* =========================================================
   SCROLLBAR
========================================================= */
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

/* =========================================================
   NAVIGATION MENU FIX (BLACK TEXT ONLY)
========================================================= */
.nav-link {
    color: #111111 !important;
    background-color: rgba(255, 255, 255, 0.88) !important;
}
.nav-link svg, .nav-link i {
    fill: #111111 !important;
    color: #111111 !important;
}
.nav-link:hover {
    background-color: rgba(0, 0, 0, 0.06) !important;
    color: #111111 !important;
}
.nav-link-selected {
    background: linear-gradient(135deg,#F58220,#D96D12) !important;
    color: #ffffff !important;
    font-weight: 700;
    box-shadow: 0px 4px 14px rgba(245,130,32,0.35);
}
.nav-link-selected svg, .nav-link-selected i {
    fill: #ffffff !important;
    color: #ffffff !important;
}

</style>
""",
            unsafe_allow_html=True)

# ==========================================================
# IMPORT APP MODULES
# ==========================================================
from app.utils.state import init_session_state
from app.pages import (dashboard, client_onboarding, research, data_quality,
                       eda_lab, feature_studio, performance, policy_simulator,
                       explainability)

# ==========================================================
# INITIALIZE SESSION STATE (non-data keys only)
# ==========================================================
init_session_state()


# ==========================================================
# LOAD FULL PORTFOLIO DATA INTO SESSION STATE
# Runs once per session via the "portfolio_loaded" guard.
# Every page reads from st.session_state.portfolio_df —
# this is the single source of truth for the entire app.
# ==========================================================
@st.cache_data(show_spinner=False)
def _load_portfolio(path: str) -> pd.DataFrame:
    """Load and normalise the full loan_book.csv — cached across reruns."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Normalise string columns
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()

    # Lowercase common categoricals for consistency
    for c in ["home_ownership", "loan_purpose", "region", "email_domain_type"]:
        if c in df.columns:
            df[c] = df[c].str.lower()

    return df


if "portfolio_loaded" not in st.session_state:
    if not CSV_PATH.exists():
        st.error(f"loan_book.csv not found at: {CSV_PATH}")
        st.stop()

    with st.spinner(f"Loading portfolio data from {CSV_PATH.name}…"):
        _df = _load_portfolio(str(CSV_PATH))

    # Derive feature list — exclude metadata and target columns
    _meta = {"applicant_id_hash", "application_date", "application_dow", "set"}
    _target = next(
        (c for c in ["default_flag", "default", "loan_default", "target"]
         if c in _df.columns),
        None,
    )
    _features = [c for c in _df.columns if c not in _meta and c != _target]

    st.session_state.portfolio_df = _df
    st.session_state.features = _features
    st.session_state.target_col = _target
    st.session_state.portfolio_loaded = True  # guard — never re-runs

# ==========================================================
# SIDEBAR
# ==========================================================
eda_selected = None
eda_compare = None
eda_outliers = None

with st.sidebar:

    st.markdown('<div class="brand-wrapper">', unsafe_allow_html=True)

    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=110)
    else:
        st.error("Missing logo: app/assets/fnb_logo.png")

    st.markdown("""
        <div class="brand-title">RISKLENS</div>
        <div class="brand-subtitle">Institutional Analytics</div>
        </div>
    """,
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    selected_page = option_menu(
        menu_title=None,
        options=[
            "Executive Dashboard",
            "Client Onboarding Center",
            "Research & Methodology",
            "Data Quality Center",
            "Interactive EDA Lab",
            "Feature Engineering Studio",
            "Model Performance Center",
            "Policy Simulator",
            "Explainability Center",
        ],
        icons=[
            "speedometer2",
            "person-plus-fill",
            "book",
            "check2-circle",
            "activity",
            "sliders",
            "bar-chart-line",
            "sliders2-vertical",
            "eye",
        ],
        default_index=0,
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "transparent"
            },
            "icon": {
                "color": "#111111",
                "font-size": "15px"
            },
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "5px 10px",
                "padding": "13px 15px",
                "border-radius": "12px",
                "color": "#111111",
                "font-weight": "500",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg,#F58220,#D96D12)",
                "color": "#ffffff",
                "font-weight": "700",
                "box-shadow": "0px 4px 14px rgba(245,130,32,0.35)",
            },
        },
    )

    # EDA sidebar controls — built from the already-loaded session state df
    if selected_page == "Interactive EDA Lab" and st.session_state.get(
            "portfolio_loaded"):
        try:
            _df = st.session_state.portfolio_df
            _tgt = st.session_state.get("target_col")
            _num, _cat = eda_lab.detect_features(_df, _tgt)
            eda_selected, eda_compare, eda_outliers = eda_lab.render_sidebar_controls(
                _num, _cat)
        except Exception as e:
            st.sidebar.error(f"EDA controls error: {e}")

# ==========================================================
# PAGE ROUTING
# ==========================================================
if selected_page == "Executive Dashboard":
    dashboard.render_page()

elif selected_page == "Client Onboarding Center":
    client_onboarding.render_page()

elif selected_page == "Research & Methodology":
    research.render_page()

elif selected_page == "Data Quality Center":
    data_quality.render_page()

elif selected_page == "Interactive EDA Lab":
    if eda_selected is not None:
        eda_lab.render_page(eda_selected, eda_compare, eda_outliers)
    else:
        st.info(
            "Please verify portfolio uploads to unlock the EDA workspace environment."
        )

elif selected_page == "Feature Engineering Studio":
    feature_studio.render_page()

elif selected_page == "Model Performance Center":
    performance.render_page()

elif selected_page == "Policy Simulator":
    policy_simulator.render_page()

elif selected_page == "Explainability Center":
    explainability.render_page()
