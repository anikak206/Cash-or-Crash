import pandas as pd
import pytest

from src import config

if not config.BEST_MODEL_PATH.exists():
    pytest.skip("models/best_model.joblib not found - run `python -m src.train_model`",
                allow_module_level=True)

from src.predict import feature_columns, predict_dataframe, predict_one, prepare_features

# Customer 1 of the dataset: two months late on the latest payment (risky)
RISKY = {"LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
         "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
         "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
         "BILL_AMT5": 0, "BILL_AMT6": 0,
         "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0}

# Customer 3 of the dataset: pays every month, never late (safe)
SAFE = {"LIMIT_BAL": 90000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 2, "AGE": 34,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 29239, "BILL_AMT2": 14027, "BILL_AMT3": 13559, "BILL_AMT4": 14331,
        "BILL_AMT5": 14948, "BILL_AMT6": 15549,
        "PAY_AMT1": 1518, "PAY_AMT2": 1500, "PAY_AMT3": 1000, "PAY_AMT4": 1000,
        "PAY_AMT5": 1000, "PAY_AMT6": 5000}


def test_prepare_features_matches_training_columns():
    X = prepare_features(pd.DataFrame([RISKY]))
    assert list(X.columns) == feature_columns()
    assert X.shape == (1, 29)
    assert not X.isna().any().any()


def test_prediction_has_expected_fields():
    result = predict_one(RISKY)
    assert 0.0 <= result["default_probability"] <= 1.0
    assert result["risk_level"] in {"Low", "Medium", "High"}
    assert result["decision"] == config.risk_band(result["default_probability"])


def test_risky_customer_scores_higher_than_safe_customer():
    assert predict_one(RISKY)["default_probability"] > predict_one(SAFE)["default_probability"]


def test_batch_matches_single_predictions():
    batch = predict_dataframe(pd.DataFrame([RISKY, SAFE]))
    assert batch.iloc[0]["default_probability"] == predict_one(RISKY)["default_probability"]
    assert batch.iloc[1]["default_probability"] == predict_one(SAFE)["default_probability"]


def test_extra_columns_such_as_id_are_ignored():
    result = predict_one({**RISKY, "ID": 1, "default": 1})
    assert result["default_probability"] == predict_one(RISKY)["default_probability"]


def test_missing_fields_raise_clear_error():
    with pytest.raises(ValueError, match="Missing required fields"):
        predict_one({"LIMIT_BAL": 20000})


def test_zero_credit_limit_is_rejected():
    with pytest.raises(ValueError, match="LIMIT_BAL"):
        predict_one({**RISKY, "LIMIT_BAL": 0})
