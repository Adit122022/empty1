from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "lending_club_synthetic.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "lending_club_processed.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"

# Random Seed
RANDOM_STATE = 42

# Feature lists
NUMERIC_FEATURES = [
    'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 
    'fico_range_low', 'revol_bal', 'revol_util', 'total_acc', 
    'mort_acc', 'pub_rec_bankruptcies', 'age'
]

CATEGORICAL_FEATURES = [
    'term', 'grade', 'home_ownership', 'verification_status', 
    'purpose', 'gender', 'race'
]

TARGET = 'loan_status'
