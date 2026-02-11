from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from pathlib import Path
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Linear Regression API")

# import os
# MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/model.pkl"))

# MODEL_PATH = "/Users/tanyaagrawal/Downloads/mlops_pipeline/backened/model/model.pkl"
MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/model.pkl"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cicd-pipeline-1-bull.onrender.com"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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