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
        "http://localhost:8001",
        "https://pimapy-app.onrender.com", # Added your Streamlit app's URL
        "https://robwiederstein.github.io"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the final, production-ready model at startup
MODEL_PATH = os.getenv("MODEL_PATH", "data/07_model_output/production_model.pkl")

try:
    model = load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Could not find model at the specified path: {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"An error occurred while loading the model: {e}")

# --- THIS IS THE FIX ---
# Define the input schema using Pydantic. These field names MUST EXACTLY MATCH
# the feature names the scikit-learn model was trained on.
class ModelInput(BaseModel):
    pregnant: int
    glucose: int
    blood_pr: int
    skin_thi: int
    insulin: int
    bmi: float
    dbts_pdgr: float
    age: int

# Health check endpoint
@app.get("/health")
def health():
    """A simple endpoint to confirm the API is running."""
    return {"status": "ok", "model_loaded": model is not None}

# Prediction endpoint
@app.post("/predict")
def predict(patient_data: ModelInput):
    """
    Receives patient data, makes a prediction using the loaded model,
    and returns the prediction and its probability.
    """
    # Convert the incoming Pydantic object to a pandas DataFrame
    df = pd.DataFrame([patient_data.model_dump()])

    try:
        # Get feature names from the first step of the pipeline.
        first_step_estimator = model.steps[0][1]
        feature_names = first_step_estimator.feature_names_in_
        
        # Reorder the DataFrame to match the order the model was trained on.
        df = df[feature_names]

        # Get raw model outputs
        probability = model.predict_proba(df)[:, 1].item()
        prediction = model.predict(df)[0]

        # Format for a user-friendly response
        prediction_str = "Diabetic" if prediction == 1 else "Non-Diabetic"
        probability_str = f"{probability * 100:.1f}%"
        
        return {
            "prediction": prediction_str,
            "probability_diabetic": probability_str
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

