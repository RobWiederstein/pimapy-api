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
        "http://localhost:8001",               # Allow local development if you have a front-end
        "https://robwiederstein.github.io"     # Allow your GitHub Pages if you have one
    ],
    allow_methods=["*"],                       # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],
)

# 1) Load the final, production-ready model at startup
# This path now points to the output of your 'final_model_training' pipeline.
MODEL_PATH = os.getenv("MODEL_PATH", "data/07_model_output/production_model.pkl")

try:
    # This single 'model' object is the entire sklearn.pipeline.Pipeline
    # which includes the scaler and the classifier.
    model = load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Could not find model at the specified path: {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"An error occurred while loading the model: {e}")

# 2) Define the input schema using Pydantic for data validation.
# These field names should match the columns your model was trained on.
class ModelInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

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
        # Get the feature names from the model pipeline itself to ensure order
        # This makes the code robust to changes in column order.
        # The 'classifier' is the final step in your scikit-learn pipeline.
        feature_names = model.named_steps['classifier'].feature_names_in_
        df = df[feature_names]

        # Get raw model outputs
        probability = model.predict_proba(df)[:, 1].item() # Probability of class 1 (diabetic)
        prediction = model.predict(df)[0]

        # Format for a user-friendly response
        prediction_str = "Diabetic" if prediction == 1 else "Non-Diabetic"
        probability_str = f"{probability * 100:.1f}%"
        
        return {
            "prediction": prediction_str,
            "probability_diabetic": probability_str
        }

    except Exception as e:
        # This will catch errors during prediction and return a helpful message
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

