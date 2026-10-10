"""
Predict default risk for new customers.

Takes customers in the ORIGINAL 23-column format (the same columns as the raw CSV),
applies the same cleaning + feature engineering used in training, and returns the
default probability and the Low / Medium / High risk decision.

Use from Python:
    from src.predict import predict_one
    predict_one({"LIMIT_BAL": 20000, "SEX": 2, ...})

Use from the command line (project root):
    python -m src.predict                              # demo on 3 real customers
    python -m src.predict --csv in.csv --out out.csv   # score a whole CSV
"""
import argparse
import json
from functools import lru_cache

import joblib
import pandas as pd

from src import config
from src.data_preprocessing import clean_data, load_raw_data
from src.feature_engineering import add_features

# The 23 columns a customer record must contain
RAW_FEATURES = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


@lru_cache(maxsize=1)
def load_model():
    """Load the saved best model once and reuse it."""
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError("models/best_model.joblib not found. Run `python -m src.train_model` first.")
    return joblib.load(config.BEST_MODEL_PATH)


def model_name() -> str:
    info = config.RESULTS_DIR / "best_model_info.json"
    return json.loads(info.read_text())["best_model"] if info.exists() else type(load_model()).__name__


@lru_cache(maxsize=1)
def feature_columns() -> list:
    """The exact feature names, in the exact order, the model was trained on."""
    return list(pd.read_csv(config.PROCESSED_DIR / "X_train.csv", nrows=0).columns)


@lru_cache(maxsize=1)
def _background() -> pd.DataFrame:
    """A small reference sample of training customers (only needed for explanations)."""
    X_train = pd.read_csv(config.PROCESSED_DIR / "X_train.csv")
    return X_train.sample(50, random_state=config.RANDOM_STATE)


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw customer columns -> the 29 model features (same steps as training)."""
    missing = [c for c in RAW_FEATURES if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    df = raw[RAW_FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    if (df["LIMIT_BAL"] <= 0).any():
        raise ValueError("LIMIT_BAL (credit limit) must be greater than 0")

    df = add_features(clean_data(df))
    return df[feature_columns()]


def predict_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Score many customers. Returns one row per customer."""
    proba = load_model().predict_proba(prepare_features(raw))[:, 1]
    out = pd.DataFrame({"default_probability": proba.round(4)}, index=raw.index)
    out["decision"] = [config.risk_band(p) for p in proba]
    out["risk_level"] = out["decision"].str.split(" Risk").str[0]
    return out


def predict_one(customer: dict, explain: bool = False) -> dict:
    """Score ONE customer (a dict with the 23 raw fields)."""
    raw = pd.DataFrame([customer])
    result = predict_dataframe(raw).iloc[0]
    answer = {
        "default_probability": float(result["default_probability"]),
        "risk_level": result["risk_level"],
        "decision": result["decision"],
    }
    if explain:
        from src.explain import top_reasons          # needs the shap library
        answer["top_reasons"] = top_reasons(load_model(), _background(), prepare_features(raw), n=5)
    return answer


def main():
    parser = argparse.ArgumentParser(description="Predict credit default risk")
    parser.add_argument("--csv", help="CSV of customers (the 23 raw columns) to score")
    parser.add_argument("--out", help="where to save the scored CSV")
    args = parser.parse_args()

    if args.csv:
        raw = pd.read_csv(args.csv)
        scored = pd.concat([raw, predict_dataframe(raw)], axis=1)
        out_path = args.out or "predictions.csv"
        scored.to_csv(out_path, index=False)
        print(f"Scored {len(raw)} customers -> {out_path}")
        print(scored["risk_level"].value_counts().to_string())
    else:
        raw = load_raw_data().head(3)
        scored = predict_dataframe(raw)
        print(f"Model: {model_name()}   (demo on the first 3 customers in the raw data)\n")
        for (idx, row), actual in zip(scored.iterrows(), raw["default"]):
            print(f"Customer ID {raw.loc[idx, 'ID']}: probability {row['default_probability']:.3f} "
                  f"-> {row['decision']}   (actually defaulted: {'yes' if actual else 'no'})")


if __name__ == "__main__":
    main()
