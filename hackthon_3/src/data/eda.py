import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(file_path: str):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print("\n--- Shape ---")
    print(df.shape)
    
    print("\n--- Info ---")
    print(df.info())
    
    print("\n--- Missing Values ---")
    missing_vals = df.isnull().sum()
    print(missing_vals[missing_vals > 0])
    
    print("\n--- Target Distribution ---")
    print(df['loan_status'].value_counts(normalize=True))
    
    print("\n--- Numerical Description ---")
    print(df.describe())
    
    # Save some plots
    os.makedirs("data/processed", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='loan_status')
    plt.title("Loan Status Distribution")
    plt.savefig("data/processed/target_dist.png")
    print("\nSaved target distribution plot to data/processed/target_dist.png")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='loan_amnt', bins=30, kde=True)
    plt.title("Loan Amount Distribution")
    plt.savefig("data/processed/loan_amnt_dist.png")
    print("\nSaved loan amount distribution plot to data/processed/loan_amnt_dist.png")

if __name__ == "__main__":
    file_path = "data/raw/lending_club_synthetic.csv"
    if os.path.exists(file_path):
        perform_eda(file_path)
    else:
        print(f"File {file_path} not found. Run make_dataset.py first.")
