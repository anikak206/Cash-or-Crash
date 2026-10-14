"""
Simulation and sanity checks for the credit default system.

Run from the project root (after `python -m src.train_model`):
    python -m src.simulate

Everything is saved in results/simulation/ :
    auc_ladder.csv / .png        simple rules versus logistic regression versus the final model
    profile_risk_mix.csv / .png  how made-up customers of three behaviour types are scored
    sensitivity.csv / .png       what happens when one input is changed at a time
    edge_cases.csv               odd and invalid inputs, expected versus observed
    api_checks.csv               HTTP status codes for good and bad requests (if FastAPI is installed)
    summary.json                 counts and key numbers
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_preprocessing import load_processed_data
from src.generate_synthetic import generate
from src.predict import RAW_FEATURES, load_model, model_name, predict_dataframe, predict_one

OUT = config.RESULTS_DIR / "simulation"
LOW, MED, HIGH = config.risk_band(0.0), config.risk_band(config.LOW_RISK_MAX), config.risk_band(config.HIGH_RISK_MIN)

# A neutral reference customer: pays on time, uses half of the limit
REFERENCE = {"LIMIT_BAL": 100000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
             "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
             **{f"BILL_AMT{i}": 50000 for i in range(1, 7)}, **{f"PAY_AMT{i}": 5000 for i in range(1, 7)}}


def ks(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr - fpr))


# ------------------------------------------------------------------ A. AUC ladder
def auc_ladder(summary):
    X_train, X_test, y_train, y_test = load_processed_data()
    steps = {
        "Guess (same score for everyone)": np.zeros(len(y_test)),
        "Rule: low credit limit": -X_test["LIMIT_BAL"].to_numpy(),
        "Rule: high credit utilisation": X_test["credit_utilization"].to_numpy(),
        "Rule: latest payment delay (PAY_0)": X_test["PAY_0"].to_numpy(),
        "Rule: worst delay in six months": X_test["max_repayment_delay"].to_numpy(),
    }
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    steps["Logistic regression, 29 features"] = lr.predict_proba(X_test)[:, 1]
    steps[f"Final model ({model_name()})"] = load_model().predict_proba(X_test)[:, 1]

    rows = [{"step": name, "roc_auc": round(float(roc_auc_score(y_test, s)), 4), "ks": round(ks(y_test, s), 4)}
            for name, s in steps.items()]
    ladder = pd.DataFrame(rows)
    ladder.to_csv(OUT / "auc_ladder.csv", index=False)

    flagged = X_test["PAY_0"] >= 2                     # a one-line business rule
    tp = int((flagged & (y_test == 1)).sum())
    summary["rule_pay0_at_least_2"] = {
        "customers_flagged_share": round(float(flagged.mean()), 4),
        "precision": round(tp / max(int(flagged.sum()), 1), 4),
        "recall": round(tp / int((y_test == 1).sum()), 4)}

    plt.figure(figsize=(8, 4.5))
    plt.barh(ladder["step"][::-1], ladder["roc_auc"][::-1], color="#4C72B0")
    plt.axvline(0.5, color="grey", ls="--", label="guessing (0.5)")
    plt.axvline(0.75, color="red", ls="--", label="requirement NFR-01 (0.75)")
    plt.xlim(0.4, 0.85); plt.xlabel("ROC-AUC on the 6,000 test customers"); plt.legend(loc="upper right")
    plt.title("From simple rules to the final model")
    plt.tight_layout(); plt.savefig(OUT / "auc_ladder.png", dpi=120); plt.close()
    summary["auc_ladder"] = {r["step"]: r["roc_auc"] for r in rows}


# ------------------------------------------------------------------ B. behaviour profiles
def profile_mix(summary):
    customers = generate(1000, seed=config.RANDOM_STATE)
    scored = predict_dataframe(customers.drop(columns="profile"))
    df = pd.concat([customers["profile"], scored], axis=1)
    out = []
    for p in ["reliable", "stretched", "distressed"]:
        sub = df[df["profile"] == p]
        out.append({"profile": p, "customers": len(sub),
                    "mean_probability": round(float(sub["default_probability"].mean()), 3),
                    "share_low": round(float((sub["decision"] == LOW).mean()), 3),
                    "share_medium": round(float((sub["decision"] == MED).mean()), 3),
                    "share_high": round(float((sub["decision"] == HIGH).mean()), 3)})
    mix = pd.DataFrame(out)
    mix.to_csv(OUT / "profile_risk_mix.csv", index=False)

    plt.figure(figsize=(7, 3.8))
    left = np.zeros(len(mix))
    for col, color, label in (("share_low", "#55A868", "Low"), ("share_medium", "#DD8452", "Medium"),
                              ("share_high", "#C44E52", "High")):
        plt.barh(mix["profile"], mix[col], left=left, color=color, label=label)
        left += mix[col].to_numpy()
    plt.xlabel("Share of customers"); plt.title("Risk level by behaviour profile (1,000 made-up customers)")
    plt.legend(ncol=3, loc="lower center"); plt.tight_layout()
    plt.savefig(OUT / "profile_risk_mix.png", dpi=120); plt.close()
    summary["profile_mix"] = out


# ------------------------------------------------------------------ C. sensitivity
def sensitivity(summary):
    rows = []

    def sweep(name, field, values, extra=None):
        frame = pd.DataFrame([{**REFERENCE, **(extra or {}), field: v} for v in values])
        probs = predict_dataframe(frame)["default_probability"].to_numpy()
        for v, p in zip(values, probs):
            rows.append({"sweep": name, "value": v, "probability": round(float(p), 4)})
        return np.array(values), probs

    pay_values, pay_probs = sweep("latest payment delay (PAY_0)", "PAY_0", list(range(-2, 9)))
    lim_values, lim_probs = sweep("credit limit (bills fixed at 50,000)", "LIMIT_BAL",
                                  [10000, 20000, 50000, 100000, 200000, 500000, 1000000])
    # Changing one personal attribute at a time, behaviour unchanged
    base = float(predict_dataframe(pd.DataFrame([REFERENCE]))["default_probability"].iloc[0])
    swaps = []
    for field, values in (("SEX", [1, 2]), ("MARRIAGE", [1, 2, 3]), ("EDUCATION", [1, 2, 3, 4]),
                          ("AGE", [21, 25, 35, 50, 65, 79])):
        _, probs = sweep(f"personal attribute: {field}", field, values)
        swaps += [abs(float(p) - base) for p in probs]

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(pay_values, pay_probs, "o-", color="#C44E52")
    axes[0].set_xlabel("PAY_0 (months late; -2 and -1 = paid on time)"); axes[0].set_ylabel("Predicted default probability")
    axes[0].set_title("Effect of the latest payment delay")
    axes[1].semilogx(lim_values, lim_probs, "o-", color="#4C72B0")
    axes[1].set_xlabel("Credit limit (bills fixed at 50,000)"); axes[1].set_title("Effect of the credit limit")
    plt.tight_layout(); plt.savefig(OUT / "sensitivity.png", dpi=120); plt.close()

    rho = float(spearmanr(pay_values, pay_probs)[0])
    summary["sensitivity"] = {
        "reference_probability": round(base, 4),
        "pay0_probability_at_minus2": round(float(pay_probs[0]), 4),
        "pay0_probability_at_0": round(float(pay_probs[2]), 4),
        "pay0_probability_at_2": round(float(pay_probs[4]), 4),
        "pay0_probability_at_8": round(float(pay_probs[-1]), 4),
        "pay0_rank_correlation_with_probability": round(rho, 3),
        "pay0_steps_that_do_not_decrease": f"{int((np.diff(pay_probs) >= -1e-9).sum())} of {len(pay_probs) - 1}",
        "limit_10k_vs_1m_probability": [round(float(lim_probs[0]), 4), round(float(lim_probs[-1]), 4)],
        "largest_change_from_one_personal_attribute": round(max(swaps), 4)}


# ------------------------------------------------------------------ D. edge cases
def _try(customer):
    try:
        r = predict_one(customer)
        return {"ok": True, "probability": r["default_probability"], "decision": r["decision"]}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {' '.join(str(e).split())[:90]}"}


def edge_cases(summary):
    def mk(**kw):
        return {**REFERENCE, **kw}

    quiet = {**{f"PAY_{k}": -2 for k in ("0", "2", "3", "4", "5", "6")},
             **{f"BILL_AMT{i}": 0 for i in range(1, 7)}, **{f"PAY_AMT{i}": 0 for i in range(1, 7)}}
    ontime = {**{f"PAY_{k}": -1 for k in ("0", "2", "3", "4", "5", "6")},
              **{f"BILL_AMT{i}": 20000 for i in range(1, 7)}, **{f"PAY_AMT{i}": 20000 for i in range(1, 7)}}
    stuck = {**{f"PAY_{k}": 6 for k in ("0", "2", "3", "4", "5", "6")},
             **{f"BILL_AMT{i}": 95000 for i in range(1, 7)}, **{f"PAY_AMT{i}": 0 for i in range(1, 7)}}
    no_age = {k: v for k, v in REFERENCE.items() if k != "AGE"}
    base_p = _try(REFERENCE)["probability"]

    cases = [
        ("New customer, no history", mk(**quiet), "no crash (the answer is recorded)", lambda r: r["ok"]),
        ("Pays in full every month", mk(**ontime), "Low risk", lambda r: r["ok"] and r["decision"] == LOW),
        ("Six months late, near the limit", mk(**stuck), "Medium or High risk, never Low",
         lambda r: r["ok"] and r["decision"] != LOW),
        ("Overpaid: negative bills", mk(**{f"BILL_AMT{i}": -5000 for i in range(1, 7)}), "no crash", lambda r: r["ok"]),
        ("Very large limit, small bills", mk(LIMIT_BAL=1000000, **{f"BILL_AMT{i}": 5000 for i in range(1, 7)}), "Low risk",
         lambda r: r["ok"] and r["decision"] == LOW),
        ("Bill three times the limit", mk(LIMIT_BAL=10000, **{f"BILL_AMT{i}": 30000 for i in range(1, 7)}), "no crash",
         lambda r: r["ok"]),
        ("Credit limit of 0", mk(LIMIT_BAL=0), "clear error mentioning LIMIT_BAL",
         lambda r: (not r["ok"]) and "LIMIT_BAL" in r["error"]),
        ("Missing field AGE", no_age, "clear error mentioning missing fields",
         lambda r: (not r["ok"]) and "Missing required fields" in r["error"]),
        ("Text instead of a number", mk(LIMIT_BAL="abc"), "error, no wrong answer", lambda r: not r["ok"]),
        ("Unknown education code 6", mk(EDUCATION=6), "same answer as code 4 (cleaning groups them)",
         lambda r: r["ok"] and r["probability"] == _try(mk(EDUCATION=4))["probability"]),
        ("Unknown marriage code 0", mk(MARRIAGE=0), "same answer as code 3 (cleaning groups them)",
         lambda r: r["ok"] and r["probability"] == _try(mk(MARRIAGE=3))["probability"]),
        ("Extra fields ID and default", {**REFERENCE, "ID": 1, "default": 1}, "extra fields ignored",
         lambda r: r["ok"] and r["probability"] == base_p),
    ]
    rows = []
    for name, customer, expected, check in cases:
        r = _try(customer)
        observed = (f"probability {r['probability']:.3f}, {r['decision']}" if r["ok"] else r["error"])
        rows.append({"case": name, "expected": expected, "observed": observed, "passed": bool(check(r))})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "edge_cases.csv", index=False)
    summary["edge_cases"] = {"passed": int(table["passed"].sum()), "total": len(table),
                             "failed_cases": table.loc[~table["passed"], "case"].tolist()}


# ------------------------------------------------------------------ E. API checks
def api_checks(summary):
    try:
        from fastapi.testclient import TestClient
        from app.app import EXAMPLE_CUSTOMER, app
    except Exception as e:
        summary["api_checks"] = {"skipped": f"{type(e).__name__}: {str(e)[:80]}"}
        return
    client = TestClient(app)
    no_age = {k: v for k, v in EXAMPLE_CUSTOMER.items() if k != "AGE"}
    checks = [
        ("GET /health", lambda: client.get("/health"), 200),
        ("POST /predict, valid customer", lambda: client.post("/predict", json=EXAMPLE_CUSTOMER), 200),
        ("POST /predict, SEX = 5", lambda: client.post("/predict", json={**EXAMPLE_CUSTOMER, "SEX": 5}), 422),
        ("POST /predict, LIMIT_BAL = 0", lambda: client.post("/predict", json={**EXAMPLE_CUSTOMER, "LIMIT_BAL": 0}), 422),
        ("POST /predict, AGE missing", lambda: client.post("/predict", json=no_age), 422),
        ("POST /predict/batch, 3 customers",
         lambda: client.post("/predict/batch", json={"customers": [EXAMPLE_CUSTOMER] * 3}), 200),
        ("POST /predict/batch, empty list", lambda: client.post("/predict/batch", json={"customers": []}), 422),
        ("GET /unknown-page", lambda: client.get("/unknown-page"), 404),
    ]
    rows = []
    for name, call, expected in checks:
        status = call().status_code
        rows.append({"request": name, "expected_status": expected, "observed_status": status, "passed": status == expected})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "api_checks.csv", index=False)
    summary["api_checks"] = {"passed": int(table["passed"].sum()), "total": len(table)}


def main():
    if not config.BEST_MODEL_PATH.exists():
        raise SystemExit("models/best_model.joblib not found. Run `python -m src.train_model` first.")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"model": model_name()}
    for step in (auc_ladder, profile_mix, sensitivity, edge_cases, api_checks):
        print(f"Running {step.__name__} ...")
        step(summary)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved results in {OUT}")
    print("AUC ladder:", summary["auc_ladder"])
    print("Edge cases:", summary["edge_cases"])
    print("API checks:", summary["api_checks"])


if __name__ == "__main__":
    main()
