import pandas as pd
import numpy as np
from typing import Optional

def generate_synthetic_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset mimicking the Lending Club structure for testing.
    """
    np.random.seed(random_state)
    
    data = {
        'loan_amnt': np.random.randint(1000, 40000, n_samples),
        'int_rate': np.random.uniform(5.0, 30.0, n_samples),
        'installment': np.random.uniform(100, 1500, n_samples),
        'annual_inc': np.random.normal(60000, 20000, n_samples),
        'dti': np.random.uniform(0, 40, n_samples),
        'fico_range_low': np.random.randint(600, 850, n_samples),
        'revol_bal': np.random.randint(0, 50000, n_samples),
        'revol_util': np.random.uniform(0, 100, n_samples),
        'total_acc': np.random.randint(5, 50, n_samples),
        'mort_acc': np.random.randint(0, 5, n_samples),
        'pub_rec_bankruptcies': np.random.choice([0, 1, 2], p=[0.9, 0.08, 0.02], size=n_samples),
        'term': np.random.choice([' 36 months', ' 60 months'], n_samples),
        'grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_samples),
        'home_ownership': np.random.choice(['RENT', 'MORTGAGE', 'OWN'], n_samples),
        'verification_status': np.random.choice(['Verified', 'Source Verified', 'Not Verified'], n_samples),
        'purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 'other'], n_samples),
        # Protected attributes for fairness analysis
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n_samples),
        # Target variable (imbalanced)
        'loan_status': np.random.choice(['Fully Paid', 'Charged Off'], p=[0.8, 0.2], size=n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some correlations for realism
    # Higher interest rate for keeping higher risk
    df.loc[df['grade'].isin(['E', 'F', 'G']), 'int_rate'] += 5.0
    df.loc[df['fico_range_low'] < 660, 'int_rate'] += 3.0
    
    # Defaults are more likely with high DTI and low FICO
    risk_score = (df['dti'] * 0.5) + (850 - df['fico_range_low']) * 0.1 + (df['int_rate'] * 2)
    threshold = np.percentile(risk_score, 80)
    df['loan_status'] = np.where(risk_score > threshold, 'Charged Off', 'Fully Paid')
    
    # Add some noise to target to make it not perfectly separable
    noise_indices = np.random.choice(df.index, size=int(n_samples * 0.1), replace=False)
    df.loc[noise_indices, 'loan_status'] = np.random.choice(['Fully Paid', 'Charged Off'], size=len(noise_indices))

    return df

if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)
    
    print("Generating synthetic data...")
    df = generate_synthetic_data(n_samples=50000)
    output_path = "data/raw/lending_club_synthetic.csv"
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(df.head())
    print(df['loan_status'].value_counts(normalize=True))
