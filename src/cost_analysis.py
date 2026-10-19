"""
Option C - Adaptive Risk Decision Engine (cost-based threshold analysis).

Question: if a missed defaulter and a wrongly flagged good customer cost different
amounts, which decision threshold gives the lowest total cost?

Run from the project root:
    python -m src.cost_analysis
    python -m src.cost_analysis --cost-default 8000 --cost-false-alarm 400

Saves to results/cost_analysis/:
    cost_sweep.csv, ratio_sensitivity.csv, summary.json, cost_curve.png

IMPORTANT: the costs are ASSUMED example values, not real bank figures.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data_preprocessing import load_processed_data

OUT_DIR = config.RESULTS_DIR / "cost_analysis"
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)

# Illustrative assumptions (currency units per customer)
DEFAULT_COST_MISSED_DEFAULTER = 2500   # loss when a customer who will default is approved
DEFAULT_COST_FALSE_ALARM = 500         # profit lost when a reliable customer is flagged/rejected


def confusion_at(y_true, proba, threshold):
    """Counts when customers with probability >= threshold are flagged as 'will default'."""
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(proba) >= threshold
    tp = int(np.sum(flagged & y_true))
    fp = int(np.sum(flagged & ~y_true))
    fn = int(np.sum(~flagged & y_true))
    tn = int(np.sum(~flagged & ~y_true))
    return tp, fp, fn, tn


def evaluate_threshold(y_true, proba, threshold, cost_default, cost_false_alarm) -> dict:
    tp, fp, fn, tn = confusion_at(y_true, proba, threshold)
    n = tp + fp + fn + tn
    return {
        "threshold": float(threshold),
        "flagged_share": (tp + fp) / n,
        "defaulters_caught_recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "cost_missed_defaulters": fn * cost_default,
        "cost_false_alarms": fp * cost_false_alarm,
        "total_cost": fn * cost_default + fp * cost_false_alarm,
    }


def sweep(y_true, proba, cost_default, cost_false_alarm, thresholds=THRESHOLDS) -> pd.DataFrame:
    return pd.DataFrame([evaluate_threshold(y_true, proba, t, cost_default, cost_false_alarm)
                         for t in thresholds])


def best_row(table: pd.DataFrame) -> pd.Series:
    """Row with the lowest total cost (ties -> the lower threshold)."""
    return table.loc[table["total_cost"].idxmin()]


def ratio_sensitivity(y_true, proba, cost_false_alarm, ratios=(1, 2, 5, 10, 20)) -> pd.DataFrame:
    """How the best threshold moves as a missed defaulter gets more expensive."""
    rows = []
    for r in ratios:
        best = best_row(sweep(y_true, proba, cost_false_alarm * r, cost_false_alarm))
        rows.append({"missed_to_false_alarm_ratio": r, "best_threshold": best["threshold"],
                     "recall": best["defaulters_caught_recall"],
                     "flagged_share": best["flagged_share"], "total_cost": best["total_cost"]})
    return pd.DataFrame(rows)


def load_test_scores():
    """Default probabilities of the saved best model on the held-out TEST set."""
    from src.predict import load_model                      # imported here to keep module import light
    _, X_test, _, y_test = load_processed_data()
    y = np.asarray(y_test).ravel()
    return y, load_model().predict_proba(X_test)[:, 1]


def run(cost_default=DEFAULT_COST_MISSED_DEFAULTER, cost_false_alarm=DEFAULT_COST_FALSE_ALARM) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y, proba = load_test_scores()

    table = sweep(y, proba, cost_default, cost_false_alarm)
    best = best_row(table)
    approve_all = evaluate_threshold(y, proba, 1.01, cost_default, cost_false_alarm)   # flags nobody
    reject_all = evaluate_threshold(y, proba, 0.0, cost_default, cost_false_alarm)     # flags everybody
    current_low = evaluate_threshold(y, proba, config.LOW_RISK_MAX, cost_default, cost_false_alarm)
    current_high = evaluate_threshold(y, proba, config.HIGH_RISK_MIN, cost_default, cost_false_alarm)
    sens = ratio_sensitivity(y, proba, cost_false_alarm)

    table.to_csv(OUT_DIR / "cost_sweep.csv", index=False)
    sens.to_csv(OUT_DIR / "ratio_sensitivity.csv", index=False)

    summary = {
        "assumption_note": "Costs are assumed example values, not real bank figures.",
        "cost_per_missed_defaulter": cost_default,
        "cost_per_false_alarm": cost_false_alarm,
        "n_customers": int(len(y)),
        "best_threshold": float(best["threshold"]),
        "best_total_cost": float(best["total_cost"]),
        "best_recall": round(float(best["defaulters_caught_recall"]), 3),
        "best_flagged_share": round(float(best["flagged_share"]), 3),
        "cost_approve_everyone": float(approve_all["total_cost"]),
        "cost_reject_everyone": float(reject_all["total_cost"]),
        f"cost_at_{config.LOW_RISK_MAX}": float(current_low["total_cost"]),
        f"cost_at_{config.HIGH_RISK_MIN}": float(current_high["total_cost"]),
        "saving_vs_approve_everyone_pct": round(
            100 * (1 - best["total_cost"] / approve_all["total_cost"]), 1),
        "saving_vs_reject_everyone_pct": round(
            100 * (1 - best["total_cost"] / reject_all["total_cost"]), 1),
        "saving_vs_current_high_cutoff_pct": round(
            100 * (1 - best["total_cost"] / current_high["total_cost"]), 1),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(table["threshold"], table["total_cost"] / 1e6, label="Total cost", color="black")
    ax.plot(table["threshold"], table["cost_missed_defaulters"] / 1e6, "--", label="Missed defaulters")
    ax.plot(table["threshold"], table["cost_false_alarms"] / 1e6, "--", label="False alarms")
    ax.axvline(best["threshold"], color="green", alpha=0.6, label=f"Best threshold = {best['threshold']:.2f}")
    ax.set_xlabel("Decision threshold (flag customer if probability >= threshold)")
    ax.set_ylabel("Cost on test set (millions, assumed units)")
    ax.set_title(f"Cost vs threshold (missed = {cost_default}, false alarm = {cost_false_alarm})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cost_curve.png", dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print("\nBest threshold as a missed defaulter gets more expensive:")
    print(sens.round(3).to_string(index=False))
    print(f"\nSaved to {OUT_DIR}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Cost-based threshold analysis")
    p.add_argument("--cost-default", type=float, default=DEFAULT_COST_MISSED_DEFAULTER,
                   help="cost of approving a customer who then defaults")
    p.add_argument("--cost-false-alarm", type=float, default=DEFAULT_COST_FALSE_ALARM,
                   help="cost of flagging/rejecting a customer who would have paid")
    a = p.parse_args()
    run(a.cost_default, a.cost_false_alarm)


if __name__ == "__main__":
    main()
