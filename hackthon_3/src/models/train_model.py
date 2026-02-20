import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import sys
import os
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, MODELS_DIR, RANDOM_STATE, MODEL_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from src.features.transformers import FeatureEngineer, OutlierHandler

# Try importing XGBoost and LightGBM
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("XGBoost not installed.")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    print("LightGBM not installed.")

def train_full_pipeline():
    print("Loading RAW data...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Data not found at {RAW_DATA_PATH}. Run make_dataset.py first.")
        return

    df = pd.read_csv(RAW_DATA_PATH)
    
    # Define new feature names that FeatureEngineer adds
    new_features = [
        'income_to_loan_ratio', 'payment_income_ratio', 'balance_income_ratio', 
        'accounts_per_age', 'internal_risk_score', 'log_annual_inc', 
        'log_revol_bal', 'log_loan_amnt', 'fico_x_income', 'int_rate_x_loan', 
        'dti_squared'
    ]
    all_numeric = NUMERIC_FEATURES + new_features
    
    # Define Preprocessing Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('outliers', OutlierHandler()),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, all_numeric),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    
    # Split data
    X = df.drop(columns=[TARGET])
    y = df[TARGET].apply(lambda x: 1 if x == 'Charged Off' else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # We need to construct the pipeline carefully. 
    # FeatureEngineer output is a DataFrame (with new columns).
    # ColumnTransformer needs to act on that DataFrame.
    # So: FE -> Preprocessor -> Model
    
    # Define models to train
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE) if xgb else None,
        'LightGBM': lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1) if lgb else None
    }
    
    best_auc = 0
    best_model_name = ""
    best_pipeline = None
    
    print("\nTraining models with full pipeline...")
    
    for name, model in models.items():
        if model is None: continue
        
        print(f"Training {name}...")
        start_time = time.time()
        
        full_pipeline = Pipeline(steps=[
            ('feature_engineering', FeatureEngineer()),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Fit
        full_pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = full_pipeline.predict(X_test)
        y_proba = full_pipeline.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"  Accuracy: {acc:.4f}, AUC: {auc:.4f}, Time: {time.time() - start_time:.2f}s")
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_pipeline = full_pipeline
            
    print(f"\nBest Model: {best_model_name} with AUC: {best_auc:.4f}")
    
    # Save best pipeline
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Saved best full pipeline to {MODEL_PATH}")
    
    # Save test data for XAI (raw form)
    test_data = X_test.copy()
    test_data['target'] = y_test
    test_data.to_csv(os.path.dirname(PROCESSED_DATA_PATH) + "/test_data_raw.csv", index=False)
    print("Saved raw test data for XAI.")

if __name__ == "__main__":
    train_full_pipeline()
