# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware 
import pandas as pd
from joblib import load
import os

app = FastAPI(title="Pima Diabetes Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",               # local dev
        "https://robwiederstein.github.io"     # your GitHub Pages root
    ],
    allow_methods=["*"],                       # GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

# 1) load the saved model at startup
MODEL_PATH = os.getenv("MODEL_PATH", "data/06_model/tuned_rf_model.pkl")
try:
    model = load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model: {e}")

# 2) define input schema
class Patient(BaseModel):
    pregnant: float
    glucose:  float
    blood_pr: float
    skin_thi: float
    insulin:  float
    bmi:      float
    dbts_pdgr:   float
    flag_imp: int
    age:      float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(patient: Patient):
    df = pd.DataFrame([patient.dict()])
    # reorder to match training
    try:
        df = df[model.feature_names_in_]
    except AttributeError:
        df = df[model.named_steps["scaler"].feature_names_in_]
    # get raw outputs
    prob = model.predict_proba(df)[:, 1].item()
    label = int(prob > 0.5)
    # format for humans
    prediction_str = "diabetic" if label == 1 else "non-diabetic"
    probability_str = f"{prob * 100:.1f}%"
    return {
        "prediction": prediction_str,
        "probability": probability_str
    }
