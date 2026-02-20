import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bound = {}
        self.upper_bound = {}
        
    def fit(self, X, y=None):
        # Handle numpy array vs dataframe
        if hasattr(X, "select_dtypes"):
            cols = X.select_dtypes(include=[np.number]).columns
            self.cols_idx = cols # store column names
            # Calculate bounds
            for col in cols:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.lower_bound[col] = Q1 - (self.factor * IQR)
                self.upper_bound[col] = Q3 + (self.factor * IQR)
        else:
            # Assume all columns are numeric if numpy array
            self.cols_idx = range(X.shape[1])
            for col_idx in self.cols_idx:
                series = pd.Series(X[:, col_idx])
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                self.lower_bound[col_idx] = Q1 - (self.factor * IQR)
                self.upper_bound[col_idx] = Q3 + (self.factor * IQR)
                
        return self

    def transform(self, X):
        X_copy = X.copy()
        if hasattr(X, "columns"):
             for col, lower in self.lower_bound.items():
                if col in X_copy.columns:
                    X_copy[col] = np.clip(X_copy[col], lower, self.upper_bound[col])
        else:
             for col_idx, lower in self.lower_bound.items():
                 X_copy[:, col_idx] = np.clip(X_copy[:, col_idx], lower, self.upper_bound[col_idx])
                 
        return X_copy

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        # 1. Income to Loan Ratio
        X_copy['income_to_loan_ratio'] = X_copy['annual_inc'] / (X_copy['loan_amnt'] + 1)
        
        # 2. Payment to Income Ratio (Monthly)
        X_copy['payment_income_ratio'] = (X_copy['installment'] * 12) / (X_copy['annual_inc'] + 1)
        
        # 3. Credit Utilization interactions
        X_copy['balance_income_ratio'] = X_copy['revol_bal'] / (X_copy['annual_inc'] + 1)
        
        # 4. Total Accounts per Year of Age (proxy for credit activity intensity)
        X_copy['accounts_per_age'] = X_copy['total_acc'] / (X_copy['age'] + 1)
        
        # 5. Risk Score Proxy
        X_copy['internal_risk_score'] = (X_copy['dti'] * 0.5) + (850 - X_copy['fico_range_low']) * 0.1

        # 6. Log Transformations for skewed features
        X_copy['log_annual_inc'] = np.log1p(X_copy['annual_inc'])
        X_copy['log_revol_bal'] = np.log1p(X_copy['revol_bal'])
        X_copy['log_loan_amnt'] = np.log1p(X_copy['loan_amnt'])

        # 7. Interaction Terms
        X_copy['fico_x_income'] = X_copy['fico_range_low'] * X_copy['log_annual_inc']
        X_copy['int_rate_x_loan'] = X_copy['int_rate'] * X_copy['loan_amnt']
        X_copy['dti_squared'] = X_copy['dti'] ** 2
        
        return X_copy

def build_preprocessing_pipeline():
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
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'  # Drop original columns that are not specified (e.g., target if passed separately)
    )
    
    # Full pipeline: Feature Engineering -> Preprocessing
    # Note: FeatureEngineer adds columns, so we need to handle that. 
    # For simplicity, we'll apply FeatureEngineer first, then Preprocessor.
    # But since ColumnTransformer requires column names, we need to know new columns.
    # To keep it simple for sklearn pipeline, we might do FE outside or update the list.
    
    return preprocessor

if __name__ == "__main__":
    from src.utils.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET
    
    print("Loading raw data...")
    if not os.path.exists(RAW_DATA_PATH):
        print("Raw data not found. Run make_dataset.py first.")
        sys.exit(1)
        
    df = pd.read_csv(RAW_DATA_PATH)
    
    print("Performing feature engineering...")
    fe = FeatureEngineer()
    df_fe = fe.transform(df)
    
    print("Columns after FE:", df_fe.columns)
    
    # Update numeric features list with new features
    new_features = [
        'income_to_loan_ratio', 'payment_income_ratio', 'balance_income_ratio', 
        'accounts_per_age', 'internal_risk_score', 'log_annual_inc', 
        'log_revol_bal', 'log_loan_amnt', 'fico_x_income', 'int_rate_x_loan', 
        'dti_squared'
    ]
    all_numeric = NUMERIC_FEATURES + new_features
    
    # Re-define preprocessor with new columns
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
        ]
    )
    
    print("Fitting preprocessor...")
    X = df_fe.drop(columns=[TARGET])
    y = df_fe[TARGET].apply(lambda x: 1 if x == 'Charged Off' else 0) # Binary target
    
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names
    try:
        num_names = all_numeric
        cat_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(CATEGORICAL_FEATURES)
        feature_names = np.concatenate([num_names, cat_names])
    except Exception as e:
        print(f"Error getting feature names: {e}")
        feature_names = [f"feat_{i}" for i in range(X_processed.shape[1])]
    
    # Save processed data
    print("Saving processed data...")
    df_processed = pd.DataFrame(X_processed, columns=feature_names)
    df_processed['target'] = y.values
    
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed data to {PROCESSED_DATA_PATH}")
    print("Shape:", df_processed.shape)
