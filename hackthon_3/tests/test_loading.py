import joblib
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import MODEL_PATH

def test_load_pipeline():
    print(f"Loading pipeline from {MODEL_PATH}...")
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("Pipeline loaded successfully.")
        
        # Test prediction on dummy data
        dummy_data = pd.DataFrame([{
            'loan_amnt': 10000,
            'int_rate': 10.0,
            'installment': 300.0,
            'annual_inc': 60000.0,
            'dti': 15.0,
            'fico_range_low': 700.0,
            'revol_bal': 5000.0,
            'revol_util': 50.0,
            'total_acc': 10.0,
            'mort_acc': 1.0,
            'pub_rec_bankruptcies': 0.0,
            'term': ' 36 months',
            'grade': 'B',
            'home_ownership': 'RENT',
            'verification_status': 'Verified',
            'purpose': 'debt_consolidation',
            'gender': 'Male',
            'race': 'White',
            'age': 30
        }])
        
        print("Scoring dummy data...")
        try:
            pred = pipeline.predict(dummy_data)
            proba = pipeline.predict_proba(dummy_data)
            print(f"Prediction: {pred[0]}, Probability: {proba[0]}")
        except Exception as e:
            print(f"Prediction failed: {e}")
            # Check if transformer issue
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Loading failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load_pipeline()
