from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

MODEL_VERSION = "1.0"
app = FastAPI()
model = joblib.load("churn_model.joblib")


class Customer(BaseModel):
    tenure: int
    MonthlyCharges: float
    Contract: Literal["Month-to-month", "One year", "Two year"]


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(customer: Customer):
    row = pd.DataFrame([customer.model_dump()])
    score = model.predict_proba(row)[0, 1]
    return {"churn_score": round(float(score), 3), "flag": bool(score >= 0.5)}
