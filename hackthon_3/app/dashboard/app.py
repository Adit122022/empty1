import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import PROCESSED_DATA_PATH, MODEL_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET

st.set_page_config(
    page_title="Credit Scoring AI",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load resources
@st.cache_resource
def load_data():
    from src.utils.config import RAW_DATA_PATH
    if os.path.exists(RAW_DATA_PATH):
        return pd.read_csv(RAW_DATA_PATH)
    return None

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def main():
    st.title("💸 Explainable AI Credit Scoring System")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Single Application", "Batch Processing", "Model Analytics", "Fairness Analysis"])
    
    data = load_data()
    model = load_model()
    
    if data is None:
        st.error("Data not found. Please run the data pipeline first.")
        return

    if model is None:
        st.warning("Model not found. Please train the model first.")

    if page == "Home":
        render_home(data, model)
    elif page == "Single Application":
        render_single_app(data, model)
    elif page == "Batch Processing":
        render_batch(data, model)
    elif page == "Model Analytics":
        render_analytics(data, model)
    elif page == "Fairness Analysis":
        render_fairness(data, model)

def render_home(data, model):
    st.markdown("### Overview")
    
    col1, col2, col3 = st.columns(3)
    
    total_loans = len(data)
    # Raw data has 'loan_status' as target, not 'target'
    # And it is string 'Charged Off' / 'Fully Paid', not 0/1
    # Debug columns if needed
    # st.write(data.columns.tolist())
    
    col1.metric("Total Applications", f"{total_loans:,}")
    
    if 'loan_status' in data.columns:
        default_rate = (data['loan_status'] == 'Charged Off').mean() * 100
    elif 'target' in data.columns:
        default_rate = data['target'].mean() * 100
    else:
        default_rate = 0.0
        st.warning(f"Target column not found. Available columns: {data.columns.tolist()}")

    col2.metric("Default Rate", f"{default_rate:.2f}%")
    
    if model:
        # Assuming model has classes_
        col3.metric("Model Status", "Active ✅")
    else:
        col3.metric("Model Status", "Not Trained ❌")
        
    st.markdown("### Recent Trends")
    # Quick chart of loan amounts
    fig = px.histogram(data, x="loan_amnt", nbins=50, title="Loan Amount Distribution")
    # Updates based on deprecation warning
    st.plotly_chart(fig) # Removed use_container_width=True to avoid warning, or use theme="streamlit" which is default

def render_single_app(data, model):
    st.header("Single Application Evaluator")
    
    with st.form("application_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amnt = st.number_input("Loan Amount", min_value=1000, max_value=40000, value=10000)
            int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=10.0)
            annual_inc = st.number_input("Annual Income", min_value=10000, max_value=1000000, value=60000)
            dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=100.0, value=15.0)
            
        with col2:
            fico = st.slider("FICO Score", 300, 850, 700)
            revol_bal = st.number_input("Revolving Balance", min_value=0, value=5000)
            total_acc = st.number_input("Total Accounts", min_value=1, value=10)
            term = st.selectbox("Term", [" 36 months", " 60 months"])
            
        submitted = st.form_submit_button("Predict")
        
    if submitted and model:
        # Construct input dataframe - this needs to match feature engineering input!
        # Ideally we reuse the pipeline. But pipeline expects raw data structure.
        # We need to recreate the raw input format.
        
        input_data = pd.DataFrame([{
            'loan_amnt': loan_amnt,
            'int_rate': int_rate,
            'annual_inc': annual_inc,
            'dti': dti,
            'fico_range_low': fico,
            'revol_bal': revol_bal,
            'total_acc': total_acc,
            'term': term,
            # Fill other required columns with defaults or user info
            'installment': (loan_amnt * (int_rate/100/12)) / (1 - (1 + int_rate/100/12)**(-36 if '36' in term else -60)), # approx
            'revol_util': 50.0, # default
            'mort_acc': 1, # default
            'pub_rec_bankruptcies': 0,
            'grade': 'B', # placeholder
            'home_ownership': 'RENT',
            'verification_status': 'Source Verified',
            'purpose': 'debt_consolidation',
            'gender': 'Male', # for fairness check
            'race': 'White',
            'age': 30
        }])
        
        # We need to run the SAME feature engineering and preprocessing steps.
        # Implies we should have saved the full pipeline including FeatureEngineer.
        # In current train_model.py, we only saved the model trained on processed data.
        # We need the preprocessor and feature engineer.
        # FIXME: train_model.py should load preprocessor from build_features execution or rebuild it.
        # For now, let's assume we can't easily predict without the full pipeline.
        
        st.warning("Prediction requires full pipeline (preprocessing + model). For this prototype, ensure `src.features.build_features` is importable and pipeline is saved.")
        
        # Placeholder prediction
        prob = np.random.uniform(0, 1) # Mock
        if prob > 0.5:
             st.success(f"Approved! Probability of Repayment: {prob:.2%}")
        else:
             st.error(f"Rejected. Probability of Repayment: {prob:.2%}")

def render_batch(data, model):
    st.header("Batch Processing")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        if st.button("Score Batch"):
            st.info("Batch scoring logic here...")

def render_analytics(data, model):
    st.header("Model Analytics")
    if os.path.exists(os.path.join(os.path.dirname(MODEL_PATH), "roc_curve.png")):
        st.image(os.path.join(os.path.dirname(MODEL_PATH), "roc_curve.png"), caption="ROC Curve")
        
    # Feature Importance
    if hasattr(model, "feature_importances_"):
        pass # Visualize

def render_fairness(data, model):
    st.header("Fairness Analysis")
    st.write("Metric: Disparate Impact Ratio")
    # Call fairness metrics here

if __name__ == "__main__":
    main()
