from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os
from pathlib import Path

app = FastAPI(title="Linear Regression API")

# ✅ Robust path (works locally, CI, Render)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "model.pkl"))

model = None

if MODEL_PATH.exists():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"⚠️ Model file not found at {MODEL_PATH}")

class InputData(BaseModel):
    area: float
    bedrooms: int

@app.get("/")
def health_check():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: InputData):

    if model is None:
        return {"error": "Model not loaded"}

    X = np.array([[data.area, data.bedrooms]])
    prediction = model.predict(X)[0]

    return {"predicted_price": float(prediction)}
