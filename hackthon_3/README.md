# 🎯 Explainable AI-Based Credit Scoring & Loan Default Prediction System

## 🌟 Project Vision

Build an enterprise-grade AI credit scoring system that doesn't just predict loan defaults, but explains every decision in human-understandable terms, ensures fairness across demographics, and provides actionable insights.

## 🏗️ Architecture

- **Data Processing**: Robust pipeline for cleaning, imputation, and feature engineering.
- **Prediction Engine**: Ensemble of models (Logistic Regression, Random Forest, XGBoost, LightGBM).
- **Explainability (XAI)**: SHAP and LIME integration for global and local explanations.
- **Fairness**: Bias detection and mitigation strategies.
- **Dashboard**: Interactive Streamlit application for users.
- **API**: FastAPI for real-time scoring and explanations.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

- **Dashboard**: `streamlit run app/dashboard/app.py`
- **API**: `uvicorn app.api.main:app --reload`

## 📂 Project Structure

- `data/`: Raw and processed datasets.
- `models/`: Trained model binaries.
- `notebooks/`: EDA and experiments.
- `src/`: Core source code.
- `app/`: Web application and API code.
