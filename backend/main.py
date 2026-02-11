from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from pathlib import Path
import numpy as np
import os

app = FastAPI(title="Linear Regression API")

import os
MODEL_PATH = Path(os.getenv("MODEL_PATH",  "backend/model/model.pkl"))


#MODEL_PATH = "C:/Users/Taha/OneDrive/Documents/mlops_pipeline/backend/model/model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

class InputData(BaseModel):
    area: float
    bedrooms: int

@app.get("/")
def health_check():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.area, data.bedrooms]])
    prediction = model.predict(X)[0]
    return {"predicted_price": float(prediction)}