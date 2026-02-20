from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import MODEL_PATH

app = FastAPI(title="Credit Scoring API", version="1.0.0")

class LoanApplication(BaseModel):
    loan_amnt: float
    int_rate: float
    installment: float
    annual_inc: float
    dti: float
    fico_range_low: float
    revol_bal: float
    revol_util: float
    total_acc: float
    mort_acc: float
    pub_rec_bankruptcies: float
    term: str
    grade: str
    home_ownership: str
    verification_status: str
    purpose: str
    gender: str = "Unknown"
    race: str = "Unknown"
    age: int = 30

@app.get("/")
def read_root():
    return {"message": "Credit Scoring API is running. Use /predict to score applications."}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}

@app.post("/predict")
def predict(application: LoanApplication):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model = joblib.load(MODEL_PATH)
        
        # Convert input to DataFrame
        data_dict = application.dict()
        df = pd.DataFrame([data_dict])
        
        # Note: If the model is just the classifier (not full pipeline), this will fail
        # because input is raw. We need the full pipeline.
        
        # For now, assuming model handles preprocessing or we have a separate preprocessor
        # (See plan to update train_model.py to save full pipeline)
        
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1] if hasattr(model, "predict_proba") else 0.0
        
        result = "Approved" if prediction == 0 else "Rejected" # 0=Fully Paid, 1=Default
        
        return {
            "decision": result,
            "probability_of_default": float(probability),
            "risk_score": int((1 - probability) * 850) # Mock score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
