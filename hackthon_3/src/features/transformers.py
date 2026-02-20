import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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
        
        # Ensure we don't divide by zero
        safe_div = lambda a, b: a / (b + 1)
        
        # 1. Income to Loan Ratio
        X_copy['income_to_loan_ratio'] = safe_div(X_copy['annual_inc'], X_copy['loan_amnt'])
        
        # 2. Payment to Income Ratio (Monthly)
        X_copy['payment_income_ratio'] = safe_div(X_copy['installment'] * 12, X_copy['annual_inc'])
        
        # 3. Credit Utilization interactions
        X_copy['balance_income_ratio'] = safe_div(X_copy['revol_bal'], X_copy['annual_inc'])
        
        # 4. Total Accounts per Year of Age
        X_copy['accounts_per_age'] = safe_div(X_copy['total_acc'], X_copy['age'])
        
        # 5. Risk Score Proxy
        X_copy['internal_risk_score'] = (X_copy['dti'] * 0.5) + (850 - X_copy['fico_range_low']) * 0.1

        # 6. Log Transformations
        X_copy['log_annual_inc'] = np.log1p(X_copy['annual_inc'])
        X_copy['log_revol_bal'] = np.log1p(X_copy['revol_bal'])
        X_copy['log_loan_amnt'] = np.log1p(X_copy['loan_amnt'])

        # 7. Interaction Terms
        X_copy['fico_x_income'] = X_copy['fico_range_low'] * X_copy['log_annual_inc']
        X_copy['int_rate_x_loan'] = X_copy['int_rate'] * X_copy['loan_amnt']
        X_copy['dti_squared'] = X_copy['dti'] ** 2
        
        return X_copy
