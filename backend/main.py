from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from pathlib import Path
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Linear Regression API")

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.pkl"

print(f"MODEL_PATH = {MODEL_PATH}")
print(" NEW VERSION MAIN.PY")


model = None

if MODEL_PATH.exists():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print("⚠️ model.pkl not found")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cicd-pipeline-1-bull.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
