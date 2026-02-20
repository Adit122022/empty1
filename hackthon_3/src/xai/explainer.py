import shap
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import PROCESSED_DATA_PATH, MODEL_PATH, DATA_DIR, CATEGORICAL_FEATURES

class ModelExplainer:
    def __init__(self, model_path=None, data_path=None):
        self.model_path = model_path or MODEL_PATH
        self.data_path = data_path or (DATA_DIR / "processed" / "lending_club_processed.csv")
        self.model = None
        self.explainer_shap = None
        self.explainer_lime = None
        self.X_train = None
        self.feature_names = None

    def load_resources(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"Loading model from {self.model_path}...")
        self.model = joblib.load(self.model_path)
        
        print("Loading background data for explainers...")
        # Load a sample of data for background distribution (SHAP/LIME need this)
        df = pd.read_csv(self.data_path)
        if 'target' in df.columns:
            self.X_train = df.drop(columns=['target'])
        else:
            self.X_train = df
            
        self.feature_names = self.X_train.columns.tolist()
        
        # Initialize SHAP Explainer
        # Check model type to choose correct explainer
        # If pipeline, extract the classifier
        if hasattr(self.model, 'predict_proba'):
             # It's a pipeline or model with predict_proba
             # We will use KernelExplainer for the full pipeline to be safe and simple
             # But KernelExplainer is slow. 
             # Let's try to see if we can extract the classifier.
             if isinstance(self.model, int): # Dummy check, actually checking steps
                 pass
             
        # Check if it is a Pipeline
        is_pipeline = hasattr(self.model, "steps") or hasattr(self.model, "named_steps")
        
        if is_pipeline:
            print("Detected Pipeline model.")
            # Verify if it has our structure: feature_engineering -> preprocessor -> classifier
            try:
                self.classifier = self.model.named_steps['classifier']
                self.preprocessor = self.model.named_steps['preprocessor']
                self.fe = self.model.named_steps['feature_engineering']
                
                # Transform background data for TreeExplainer
                # We need to transform X_train (raw) -> X_train (processed)
                print("Transforming background data for SHAP...")
                X_fe = self.fe.transform(self.X_train)
                X_transformed = self.preprocessor.transform(X_fe)
                
                # Get feature names from preprocessor
                try:
                    # Generic way to get feature names from ColumnTransformer
                    # valid for sklearn > 1.0 (get_feature_names_out)
                    self.feature_names = self.preprocessor.get_feature_names_out()
                except:
                    self.feature_names = [f"feat_{i}" for i in range(X_transformed.shape[1])]
                
                self.X_background_transformed = shap.sample(X_transformed, 100)
                
                print("Initializing SHAP TreeExplainer...")
                self.explainer_shap = shap.TreeExplainer(self.classifier, data=self.X_background_transformed)
                self.use_transformed_data = True
                
            except Exception as e:
                print(f"Could not extract steps for TreeExplainer ({e}). Fallback to KernelExplainer on raw data.")
                self.explainer_shap = shap.KernelExplainer(self.model.predict_proba, shap.sample(self.X_train, 50))
                self.use_transformed_data = False
        else:
             print("Initializing SHAP TreeExplainer on model directly...")
             self.explainer_shap = shap.TreeExplainer(self.model, data=shap.sample(self.X_train, 100))
             self.use_transformed_data = False

        print("Initializing LIME Explainer...")
        
        # 1. Identify categorical features indices
        self.categorical_features_indices = [
            i for i, col in enumerate(self.X_train.columns) 
            if col in CATEGORICAL_FEATURES or self.X_train[col].dtype == 'object'
        ]
        
        # 2. LIME requires data as numpy array. For categorical, it expects integers if we specify categorical_features?
        # Actually LimeTabularExplainer works best with label encoded data if we want to use categorical_features.
        # But our model expects RAW strings.
        # Solution: Use a custom prediction function that converts LIME's input back to DataFrame with Strings.
        # But LIME generates perturbations. If we pass strings, it can't perturb them numerically?
        # LIME supports categorical features by frequency if we pass training_data as strings?
        
        # Let's try passing raw numpy array (astype str) and letting LIME know which are categorical.
        
        self.explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.feature_names,
            class_names=['Fully Paid', 'Charged Off'],
            categorical_features=self.categorical_features_indices,
            mode='classification',
            discretize_continuous=True
        )

    def predict_fn_lime(self, X_numpy):
        # Convert numpy array back to DataFrame with correct dtypes for the pipeline
        df = pd.DataFrame(X_numpy, columns=self.feature_names)
        
        # Enforce numeric types for numeric columns (LIME might have made them objects if mixed array)
        for col in self.X_train.columns:
            if col not in CATEGORICAL_FEATURES:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return self.model.predict_proba(df)

    def explain_local(self, instance_idx=None, instance_data=None):
        """
        Explain a single instance.
        """
        if not self.explainer_shap:
            self.load_resources()
            
        if instance_data is None:
             if instance_idx is None:
                 instance_idx = 0
             instance = self.X_train.iloc[[instance_idx]]
        else:
            instance = pd.DataFrame([instance_data], columns=self.feature_names)
            
        # SHAP
        if getattr(self, 'use_transformed_data', False):
             X_fe = self.fe.transform(instance)
             instance_final = self.preprocessor.transform(X_fe)
             shap_values = self.explainer_shap.shap_values(instance_final)
        else:
             shap_values = self.explainer_shap.shap_values(instance)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        # LIME
        if instance_data is None:
            instance_exp = self.X_train.iloc[instance_idx]
        else:
            instance_exp = pd.Series(instance_data, index=self.feature_names)
            
        # Pass instance as numpy array to LIME
        lime_exp = self.explainer_lime.explain_instance(
            instance_exp.values, 
            self.predict_fn_lime, 
            num_features=5
        )
        
        return {
            "shap_values": shap_values,
            "lime_exp": lime_exp,
            "instance": instance
        }
    
    def generate_narrative(self, shap_explanation):
        # Generate text based on top SHAP values
        vals = shap_explanation['shap_values'][0]
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': vals
        }).sort_values(by='importance', key=abs, ascending=False)
        
        top_3 = feature_importance.head(3)
        
        narrative = "Analysis of Key Factors:\n"
        for _, row in top_3.iterrows():
            impact = "increases" if row['importance'] > 0 else "decreases"
            narrative += f"- {row['feature']}: {row['importance']:.2f} ({impact} risk)\n"
            
        return narrative

if __name__ == "__main__":
    # Check if raw test data exists (better for pipeline models)
    raw_test_data = DATA_DIR / "processed" / "test_data_raw.csv"
    if os.path.exists(raw_test_data):
        data_path = raw_test_data
    else:
        data_path = None
        
    explainer = ModelExplainer(data_path=data_path)
    explainer.load_resources()
    explainer.explain_global()
    
    # Test local explanation
    print("\nLocal Explanation for first instance:")
    result = explainer.explain_local(instance_idx=0)
    print(explainer.generate_narrative(result))
