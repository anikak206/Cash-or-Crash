"""
Evaluate the saved best model on the held-out TEST set.

Run from the project root (after `python -m src.train_model`):
    python -m src.evaluate

Saves to results/:  metrics.json, threshold_table.csv, risk_band_summary.csv
and to results/plots/:  roc_curve, pr_curve, calibration_curve, confusion_matrix, risk_bands
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")                       # draw to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve)

from src import config
from src.data_preprocessing import load_processed_data

PLOT_DIR = config.RESULTS_DIR / "plots"
THRESHOLDS = [0.20, 0.25, 0.30, 0.40, 0.50, 0.60]


# ---------------------------------------------------------------- metrics
def ks_statistic(y_true, proba) -> float:
    """KS = biggest gap between the 'defaulters' and 'non-defaulters' score curves."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(np.max(tpr - fpr))


def compute_metrics(y_true, proba) -> dict:
    auc = roc_auc_score(y_true, proba)
    return {
        "roc_auc": round(float(auc), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks_statistic": round(ks_statistic(y_true, proba), 4),
        "brier_score": round(float(brier_score_loss(y_true, proba)), 4),
        "default_rate_in_test": round(float(np.mean(y_true)), 4),
        "n_test_rows": int(len(y_true)),
    }


def threshold_table(y_true, proba) -> pd.DataFrame:
    """Precision / recall at several cut-offs (higher cut-off = fewer customers flagged)."""
    y_true = np.asarray(y_true)
    rows = []
    for t in THRESHOLDS:
        flagged = proba >= t
        tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"threshold": t, "flagged_share": flagged.mean(), "precision": precision,
                     "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return pd.DataFrame(rows).round(3)


def risk_band_summary(y_true, proba) -> pd.DataFrame:
    """How many customers land in each band, and how many of them REALLY defaulted."""
    order = [config.risk_band(0.0), config.risk_band(config.LOW_RISK_MAX),
             config.risk_band(config.HIGH_RISK_MIN)]
    df = pd.DataFrame({"band": [config.risk_band(p) for p in proba], "default": np.asarray(y_true)})
    out = df.groupby("band")["default"].agg(customers="count", actual_default_rate="mean")
    out = out.reindex(order).fillna(0)
    out["share_of_customers"] = out["customers"] / out["customers"].sum()
    return out.round(3).reset_index()


# ---------------------------------------------------------------- plots
def _save(name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{name}.png", dpi=120)
    plt.close()


def make_plots(y_true, proba, metrics, bands):
    fpr, tpr, _ = roc_curve(y_true, proba)
    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="random guess")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curve (test set)"); plt.legend()
    _save("roc_curve")

    prec, rec, _ = precision_recall_curve(y_true, proba)
    plt.figure(figsize=(5.5, 5))
    plt.plot(rec, prec, label=f"PR-AUC = {metrics['pr_auc']:.3f}")
    plt.axhline(np.mean(y_true), ls="--", color="grey", label="no-skill (default rate)")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-recall curve (test set)"); plt.legend()
    _save("pr_curve")

    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5.5, 5))
    plt.plot(mean_pred, frac_pos, "o-", label="model")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    plt.xlabel("Predicted default probability"); plt.ylabel("Actual default rate")
    plt.title("Calibration curve (test set)"); plt.legend()
    _save("calibration_curve")

    t = config.LOW_RISK_MAX
    ConfusionMatrixDisplay(confusion_matrix(y_true, proba >= t),
                           display_labels=["No default", "Default"]).plot(cmap="Blues")
    plt.title(f"Confusion matrix (flag if probability >= {t})")
    _save("confusion_matrix")

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(range(len(bands)), bands["actual_default_rate"], color=["#55A868", "#DD8452", "#C44E52"])
    plt.xticks(range(len(bands)), [b.replace(" - ", "\n") for b in bands["band"]])
    for bar, n in zip(bars, bands["customers"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"n={int(n)}", ha="center")
    plt.axhline(np.mean(y_true), ls="--", color="grey", label="overall default rate")
    plt.ylabel("Actual default rate"); plt.title("Do the risk bands separate defaulters?"); plt.legend()
    _save("risk_bands")


# ---------------------------------------------------------------- main
def main():
    if not config.BEST_MODEL_PATH.exists():
        raise SystemExit("models/best_model.joblib not found. Run `python -m src.train_model` first.")

    model = joblib.load(config.BEST_MODEL_PATH)
    _, X_test, _, y_test = load_processed_data()
    proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, proba)
    thresholds = threshold_table(y_test, proba)
    bands = risk_band_summary(y_test, proba)

    info_path = config.RESULTS_DIR / "best_model_info.json"
    metrics["model"] = json.loads(info_path.read_text())["best_model"] if info_path.exists() else type(model).__name__
    metrics["risk_thresholds"] = {"low_max": config.LOW_RISK_MAX, "high_min": config.HIGH_RISK_MIN}

    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    thresholds.to_csv(config.RESULTS_DIR / "threshold_table.csv", index=False)
    bands.to_csv(config.RESULTS_DIR / "risk_band_summary.csv", index=False)
    make_plots(y_test, proba, metrics, bands)

    print(f"Model: {metrics['model']}")
    for k in ("roc_auc", "pr_auc", "gini", "ks_statistic", "brier_score"):
        print(f"  {k:13s}: {metrics[k]}")
    print("\nPrecision / recall at different cut-offs:")
    print(thresholds.to_string(index=False))
    print("\nRisk bands (does the actual default rate rise from Low to High?):")
    print(bands.to_string(index=False))
    print(f"\nSaved metrics and plots in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
