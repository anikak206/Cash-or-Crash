"""
Measure how fast and how big the system is, and record the software it ran on.

Run from the project root (after `python -m src.train_model`):
    python -m src.benchmark

Saves results/benchmark.json. The numbers depend on the computer, so the file also records
the machine and the package versions. Re-run it on any computer to compare.
"""
import json
import os
import platform
import sys
import time
from importlib import metadata

import joblib
import numpy as np
import pandas as pd

from src import config
from src.generate_synthetic import generate
from src.predict import RAW_FEATURES, load_model, model_name, predict_dataframe, predict_one, prepare_features

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm", "shap", "joblib",
            "fastapi", "pydantic", "starlette", "uvicorn", "httpx", "pytest", "matplotlib"]


def _stats_ms(seconds: list) -> dict:
    ms = np.array(seconds) * 1000
    return {"runs": len(ms), "mean_ms": round(float(ms.mean()), 2),
            "median_ms": round(float(np.median(ms)), 2),
            "p95_ms": round(float(np.percentile(ms, 95)), 2), "max_ms": round(float(ms.max()), 2)}


def _time_calls(func, repeats: int) -> list:
    func()                                                  # warm-up call, not counted
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return times


def machine_info() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor() or "unknown", "cpu_cores": os.cpu_count(),
            "package_versions": versions}


def main():
    if not config.BEST_MODEL_PATH.exists():
        raise SystemExit("models/best_model.joblib not found. Run `python -m src.train_model` first.")

    result = {"model": model_name(), "machine": machine_info()}

    # ---- file sizes and data size
    sizes = {p.name: round(p.stat().st_size / 1e6, 2)
             for p in (config.BEST_MODEL_PATH, config.MODEL_PATH) if p.exists()}
    result["model_file_size_mb"] = sizes
    X_train = pd.read_csv(config.PROCESSED_DIR / "X_train.csv")
    result["training_data"] = {"rows": int(len(X_train)), "features": int(X_train.shape[1]),
                               "memory_mb": round(float(X_train.memory_usage(deep=True).sum() / 1e6), 2)}

    # ---- cold load of the model file
    start = time.perf_counter()
    joblib.load(config.BEST_MODEL_PATH)
    result["model_load_seconds"] = round(time.perf_counter() - start, 3)

    # ---- speed of scoring
    customers = generate(1000, seed=config.RANDOM_STATE).drop(columns="profile")
    one = customers.iloc[0].to_dict()
    load_model()
    result["single_prediction"] = _stats_ms(_time_calls(lambda: predict_one(one), 100))
    batch = {}
    for n in (1, 10, 100, 1000):
        sample = customers.head(n)
        seconds = float(np.median(_time_calls(lambda: predict_dataframe(sample), 5)))
        batch[str(n)] = {"seconds": round(seconds, 4), "ms_per_customer": round(seconds / n * 1000, 3)}
    result["batch_prediction"] = batch
    feats = prepare_features(customers.head(1))
    result["feature_preparation_ms"] = _stats_ms(_time_calls(lambda: prepare_features(customers.head(1)), 50))["median_ms"]
    result["model_only_ms_single_customer"] = _stats_ms(
        _time_calls(lambda: load_model().predict_proba(feats), 100))["median_ms"]

    # ---- explanation speed (needs shap)
    try:
        from src.explain import top_reasons
        from src.predict import _background
        row = prepare_features(customers.head(1))
        result["explanation_single_customer"] = _stats_ms(
            _time_calls(lambda: top_reasons(load_model(), _background(), row), 3))
    except Exception as e:
        result["explanation_single_customer"] = {"skipped": f"{type(e).__name__}: {str(e)[:80]}"}

    # ---- API round trip, in the same process (includes input validation)
    try:
        from fastapi.testclient import TestClient
        from app.app import EXAMPLE_CUSTOMER, app
        client = TestClient(app)
        result["api_predict_in_process"] = _stats_ms(
            _time_calls(lambda: client.post("/predict", json=EXAMPLE_CUSTOMER), 50))
    except Exception as e:
        result["api_predict_in_process"] = {"skipped": f"{type(e).__name__}: {str(e)[:80]}"}

    # ---- training times and top features, copied from earlier results
    comparison = config.RESULTS_DIR / "model_comparison.csv"
    if comparison.exists():
        c = pd.read_csv(comparison)
        result["training_seconds_incl_5fold_cv"] = {r.model: round(float(r.seconds), 1) for r in c.itertuples()}
    shap_file = config.RESULTS_DIR / "shap" / "feature_importance.csv"
    if shap_file.exists():
        s = pd.read_csv(shap_file).head(10)
        result["top_10_features_mean_abs_shap"] = {r.feature: round(float(r.mean_abs_shap), 4) for r in s.itertuples()}

    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "benchmark.json").write_text(json.dumps(result, indent=2))

    sp = result["single_prediction"]
    print(f"Model: {result['model']}   file size: {sizes}")
    print(f"Single prediction: median {sp['median_ms']} ms, 95th percentile {sp['p95_ms']} ms")
    print("Batch:", {k: f"{v['seconds']} s" for k, v in batch.items()})
    print(f"Saved {config.RESULTS_DIR / 'benchmark.json'}")


if __name__ == "__main__":
    main()
