import pytest

from src import config

pytest.importorskip("fastapi")
pytest.importorskip("httpx")        # needed by FastAPI's TestClient

if not config.BEST_MODEL_PATH.exists():
    pytest.skip("models/best_model.joblib not found - run `python -m src.train_model`",
                allow_module_level=True)

from fastapi.testclient import TestClient

from app.app import EXAMPLE_CUSTOMER, app

client = TestClient(app)


def test_health_is_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_probability_and_decision():
    response = client.post("/predict", json=EXAMPLE_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert "top_reasons" not in body            # only returned when explain=true


def test_invalid_value_is_rejected():
    response = client.post("/predict", json={**EXAMPLE_CUSTOMER, "SEX": 5})
    assert response.status_code == 422


def test_zero_credit_limit_is_rejected():
    response = client.post("/predict", json={**EXAMPLE_CUSTOMER, "LIMIT_BAL": 0})
    assert response.status_code == 422


def test_missing_field_is_rejected():
    incomplete = {k: v for k, v in EXAMPLE_CUSTOMER.items() if k != "AGE"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_batch_prediction():
    response = client.post("/predict/batch", json={"customers": [EXAMPLE_CUSTOMER, EXAMPLE_CUSTOMER]})
    assert response.status_code == 200
    assert response.json()["count"] == 2


def test_empty_batch_is_rejected():
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
