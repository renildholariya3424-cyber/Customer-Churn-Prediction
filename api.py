import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path("model")

app = FastAPI(title="Churn Prediction API", description="Predicts the probability that a telecom customer will churn")

model = joblib.load(MODEL_DIR / "churn_model.joblib") if (MODEL_DIR / "churn_model.joblib").exists() else None

YesNo = Literal["Yes", "No"]
InternetOption = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "No metrics found. Run train.py first.")
    return json.loads(path.read_text())


@app.post("/predict")
def predict(customer: Customer):
    if model is None:
        raise HTTPException(503, "Model not trained yet. Run: python train.py")
    df = pd.DataFrame([customer.model_dump()])
    prob = float(model.predict_proba(df)[0, 1])
    risk = "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"
    return {"churn_probability": round(prob, 3), "will_churn": prob >= 0.5, "risk": risk}
