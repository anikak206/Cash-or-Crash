"""
FastAPI service for credit default prediction.

Run from the project root:
    python -m uvicorn app.app:app --port 8000
Then open  http://127.0.0.1:8000/docs  to try it in the browser.

Endpoints
    GET  /health          is the service up and which model is loaded
    POST /predict         score ONE customer  (add ?explain=true for top reasons)
    POST /predict/batch   score up to 1000 customers at once
"""
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src import config
from src.predict import load_model, model_name, predict_dataframe, predict_one

app = FastAPI(
    title="Cash or Crash - Credit Default Risk API",
    version="1.0.0",
    description="Predicts the probability that a credit card customer will default "
                "and returns a Low / Medium / High risk decision.",
)

EXAMPLE_CUSTOMER = {
        "LIMIT_BAL": 20000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": -1,
        "PAY_4": -1,
        "PAY_5": -2,
        "PAY_6": -2,
        "BILL_AMT1": 3913,
        "BILL_AMT2": 3102,
        "BILL_AMT3": 689,
        "BILL_AMT4": 0,
        "BILL_AMT5": 0,
        "BILL_AMT6": 0,
        "PAY_AMT1": 0,
        "PAY_AMT2": 689,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0,
    }


class Customer(BaseModel):
    """One customer, in the same 23-column format as the raw dataset."""
    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE_CUSTOMER})

    LIMIT_BAL: float = Field(gt=0, description="Credit limit in NT dollars (must be > 0)")
    SEX: int = Field(ge=1, le=2, description="1 = male, 2 = female")
    EDUCATION: int = Field(ge=0, le=6, description="1 = graduate school, 2 = university, 3 = high school, 4 = others")
    MARRIAGE: int = Field(ge=0, le=3, description="1 = married, 2 = single, 3 = others")
    AGE: int = Field(ge=18, le=100, description="Age in years")
    PAY_0: int = Field(ge=-2, le=9, description="Repayment status, latest month (-2/-1 = paid on time, 0 = revolving credit, 1..9 = months late)")
    PAY_2: int = Field(ge=-2, le=9, description="Repayment status, 2 months ago (same scale as PAY_0)")
    PAY_3: int = Field(ge=-2, le=9, description="Repayment status, 3 months ago (same scale as PAY_0)")
    PAY_4: int = Field(ge=-2, le=9, description="Repayment status, 4 months ago (same scale as PAY_0)")
    PAY_5: int = Field(ge=-2, le=9, description="Repayment status, 5 months ago (same scale as PAY_0)")
    PAY_6: int = Field(ge=-2, le=9, description="Repayment status, 6 months ago (same scale as PAY_0)")
    BILL_AMT1: float = Field(description="Bill statement amount, month 1 (1 = most recent)")
    BILL_AMT2: float = Field(description="Bill statement amount, month 2 (1 = most recent)")
    BILL_AMT3: float = Field(description="Bill statement amount, month 3 (1 = most recent)")
    BILL_AMT4: float = Field(description="Bill statement amount, month 4 (1 = most recent)")
    BILL_AMT5: float = Field(description="Bill statement amount, month 5 (1 = most recent)")
    BILL_AMT6: float = Field(description="Bill statement amount, month 6 (1 = most recent)")
    PAY_AMT1: float = Field(ge=0, description="Amount paid, month 1 (1 = most recent)")
    PAY_AMT2: float = Field(ge=0, description="Amount paid, month 2 (1 = most recent)")
    PAY_AMT3: float = Field(ge=0, description="Amount paid, month 3 (1 = most recent)")
    PAY_AMT4: float = Field(ge=0, description="Amount paid, month 4 (1 = most recent)")
    PAY_AMT5: float = Field(ge=0, description="Amount paid, month 5 (1 = most recent)")
    PAY_AMT6: float = Field(ge=0, description="Amount paid, month 6 (1 = most recent)")


class Reason(BaseModel):
    feature: str
    value: float
    effect_on_probability: float
    direction: str


class Prediction(BaseModel):
    default_probability: float
    risk_level: str
    decision: str
    top_reasons: Optional[list[Reason]] = None


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    count: int
    predictions: list[Prediction]


@app.get("/")
def root():
    return {"service": "Cash or Crash - Credit Default Risk API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    try:
        load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "status": "ok",
        "model": model_name(),
        "risk_thresholds": {"low_below": config.LOW_RISK_MAX, "high_from": config.HIGH_RISK_MIN},
    }


@app.post("/predict", response_model=Prediction, response_model_exclude_none=True)
def predict(customer: Customer, explain: bool = False):
    try:
        return predict_one(customer.model_dump(), explain=explain)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    try:
        raw = pd.DataFrame([c.model_dump() for c in request.customers])
        scored = predict_dataframe(raw)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    rows = scored[["default_probability", "risk_level", "decision"]].to_dict(orient="records")
    return {"count": len(rows), "predictions": rows}
