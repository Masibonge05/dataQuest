# RiskLens Analytics

RiskLens Analytics is a credit risk analytics and portfolio simulation platform built for the FNB DataQuest 2026 Competition. The platform provides interactive dashboards and machine learning tools for analyzing credit applications, evaluating risk models, monitoring portfolio performance, and simulating lending policies.

## Features
- Credit risk dashboard
- Data quality and drift analysis
- Exploratory data analysis (EDA)
- WoE and IV feature engineering
- Model performance evaluation
- Lending policy simulation
- Explainable AI insights

## Tech Stack
- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- Google Gemini AI

## Running the Project

### Install dependencies
```bash
pip install -r requirements.txt
```

### Create Gemini API Key

1. Go to Google AI Studio:
https://aistudio.google.com/

2. Sign in with your Google account.

3. Click **"Get API Key"**.

4. Create a new API key and copy it.

### Configure Gemini API Key

Create a `.env` file in the root folder of the project and add:

```env
GEMINI_API_KEY=your_api_key_here
```

Example:

```env
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXX
```

### Start the application
```bash
streamlit run main.py
```

The app will run on:

```bash
http://localhost:8501
```

# dataQuest