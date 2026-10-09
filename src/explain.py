"""
Explain the saved best model with SHAP.

Run from the project root (after `python -m src.train_model`):
    python -m src.explain

SHAP gives every feature a score for every customer: how many percentage points
that feature pushed the default probability UP (+) or DOWN (-).

We use SHAP's model-agnostic explainer on the probability output, so it works for
every model we train (logistic regression, forest, XGBoost, LightGBM, ensemble).

Saves to results/shap/: feature_importance.csv, summary_bar, summary_beeswarm,
waterfall_highest_risk, waterfall_lowest_risk (PNG files).
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data_preprocessing import load_processed_data

SHAP_DIR = config.RESULTS_DIR / "shap"
N_BACKGROUND = 50       # reference customers SHAP compares against
N_EXPLAIN = 200         # test customers to explain (SHAP is slow, so we sample)


def make_explainer(model, background: pd.DataFrame):
    """Build a SHAP explainer that explains the model's default PROBABILITY."""
    import shap                                    # imported here so the API can skip it

    cols = list(background.columns)

    def predict_default_proba(X):
        return model.predict_proba(pd.DataFrame(X, columns=cols))[:, 1]

    masker = shap.maskers.Independent(background, max_samples=N_BACKGROUND)
    return shap.Explainer(predict_default_proba, masker, algorithm="permutation", feature_names=cols)


def top_reasons(model, background: pd.DataFrame, customer: pd.DataFrame, n: int = 5) -> list:
    """
    Top-n reasons for ONE customer (a 1-row DataFrame), as plain dictionaries.
    Reused later by predict.py and the API for "why was this customer flagged?".
    """
    explainer = make_explainer(model, background)
    exp = explainer(customer, max_evals=2 * customer.shape[1] + 1 + 200, silent=True)
    values = exp.values[0]
    order = np.argsort(-np.abs(values))[:n]
    return [{"feature": customer.columns[i],
             "value": float(customer.iloc[0, i]),
             "effect_on_probability": round(float(values[i]), 4),
             "direction": "raises risk" if values[i] > 0 else "lowers risk"} for i in order]


def _save_current_figure(name):
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(SHAP_DIR / f"{name}.png", dpi=120, bbox_inches="tight")
    plt.close("all")


def main():
    import shap

    if not config.BEST_MODEL_PATH.exists():
        raise SystemExit("models/best_model.joblib not found. Run `python -m src.train_model` first.")

    model = joblib.load(config.BEST_MODEL_PATH)
    X_train, X_test, _, _ = load_processed_data()

    background = X_train.sample(N_BACKGROUND, random_state=config.RANDOM_STATE)
    sample = X_test.sample(min(N_EXPLAIN, len(X_test)), random_state=config.RANDOM_STATE)

    print(f"Explaining {len(sample)} customers (this takes a minute or two)...")
    explainer = make_explainer(model, background)
    exp = explainer(sample, max_evals=2 * sample.shape[1] + 1 + 200)

    # 1) Global importance table: average size of each feature's effect
    importance = (pd.DataFrame({"feature": sample.columns,
                                "mean_abs_shap": np.abs(exp.values).mean(axis=0)})
                  .sort_values("mean_abs_shap", ascending=False))
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    importance.round(5).to_csv(SHAP_DIR / "feature_importance.csv", index=False)

    # 2) Charts
    shap.plots.bar(exp, max_display=15, show=False)
    _save_current_figure("summary_bar")

    shap.plots.beeswarm(exp, max_display=15, show=False)
    _save_current_figure("summary_beeswarm")

    proba = model.predict_proba(sample)[:, 1]
    for label, idx in (("highest", int(np.argmax(proba))), ("lowest", int(np.argmin(proba)))):
        shap.plots.waterfall(exp[idx], max_display=10, show=False)
        _save_current_figure(f"waterfall_{label}_risk")
        print(f"  {label}-risk customer in sample: default probability {proba[idx]:.2f}")

    print("\nTop 10 features by average effect on default probability:")
    print(importance.head(10).round(4).to_string(index=False))
    print(f"\nSaved to {SHAP_DIR}")


if __name__ == "__main__":
    main()
