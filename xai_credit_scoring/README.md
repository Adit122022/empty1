# 🏦 FinTrust AI: Credit Intelligence Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fintrust-ai.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**FinTrust AI** is a state-of-the-art, AI-powered credit scoring platform designed to modernize traditional lending decisions. By combining sophisticated machine learning ensembles with **Explainable AI (XAI)** and **Responsible AI** principles, FinTrust provides banks and financial institutions with a transparent, fair, and high-accuracy risk assessment engine.

---

## 🏛️ The Five Pillars of FinTrust AI

Our platform is built on five core integration points that ensure reliability, transparency, and ethical decision-making.

### 1. 🧠 The Brain: Automated Model Training

- **Engine**: `GridSearchCV` pipeline in [train_model.py](train_model.py).
- **Logic**: Our system doesn't rely on a single algorithm. It automatically compares and benchmarks **XGBoost, LightGBM, Random Forest, and Gradient Boosting**.
- **Value**: "The platform features a dynamic training pipeline that auto-selects the best-performing model based on **AUC-ROC** metrics, ensuring we always use the most accurate logic for the latest demographic data."

### 2. ⚡ The Decision: Real-time Scoring

- **Engine**: Flask API ([flask_api.py](flask_api.py)) and Streamlit Inference ([app.py](app.py)).
- **Logic**: High-performance inference using `model.predict_proba()`.
- **Value**: "Decisions are made in milliseconds. Our production engine evaluates 20+ behavioral and financial data points to output a granular Probability of Default, rather than just a binary Yes/No."

### 3. 🔍 The Transparency: Explainable AI (XAI)

- **Engine**: **SHAP (SHapley Additive exPlanations)** integration.
- **Logic**: Uses `shap.TreeExplainer` to decompose the "black box" model.
- **Value**: "We prioritize transparency. Every score is accompanied by a SHAP breakdown, mathematically proving exactly which features (like credit history or checking status) most influenced the final decision."

### 4. ⚖️ The Ethics: Responsible AI & Fairness

- **Engine**: `Fairlearn` metrics integration.
- **Logic**: Real-time auditing for **Demographic Parity** and bias detection.
- **Value**: "FinTrust is built for the regulator-ready era. Our fairness auditing layer monitors model decisions for bias against protected classes, ensuring equitable access to credit for all individuals."

### 5. 🎮 The Sandbox: What-If Simulation

- **Engine**: Dynamic Inference UI in Streamlit.
- **Logic**: Real-time recalculation of risk scores based on user-controlled feature sliders.
- **Value**: "Our interactive sandbox allows underwriters and users to simulate 'What-If' scenarios—seeing exactly how a 10% increase in savings or a change in employment status would impact their future creditworthiness."

---

## 🛠️ Tech Stack

- **Frontend/Dashboard**: [Streamlit](https://streamlit.io/) (Premium Dark Theme)
- **Backend API**: [Flask](https://flask.palletsprojects.com/)
- **AI/ML Frameworks**: `scikit-learn`, `XGBoost`, `LightGBM`
- **Explainability**: `SHAP`
- **Fairness Auditing**: `Fairlearn`
- **Data Visualization**: `Plotly`, `Matplotlib`
- **External Integration**: Mock API for Credit Bureau (PAN card retrieval)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Virtual Environment (recommended)

### Installation

1.  **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/xai-credit-scoring.git
    cd xai-credit-scoring
    ```

2.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the Model**:
    ```bash
    python train_model.py
    ```

### Usage

- **Launch the UI**:
  ```bash
  streamlit run app.py
  ```
- **Run the API**:
  ```bash
  python flask_api.py
  ```

---

## 📂 Project Structure

```text
├── app.py                  # Main Streamlit Dashboard (UI & Logic)
├── train_model.py          # Model Training & Comparison Pipeline
├── flask_api.py            # RESTful API for Score Retrieval
├── pan_api_client.py       # Integration for Bureau Data Lookup
├── data/
│   └── processed_credit_data.csv  # Cleaned Dataset
├── model.pkl               # Serialized Best-Performing Model
└── requirements.txt        # Project Dependencies
```

---

## 🛡️ Responsible AI Commitment

FinTrust AI adheres to the **RBI's guidelines** for digital lending. We ensure that:

1.  Model decisions are **explainable** to the end-user.
2.  Data is fetched via **encrypted** bureau channels.
3.  Models are regularly **audited for bias** using the Fairlearn framework.

---

_Created for the FinTrust AI Credit Intelligence Presentation._
