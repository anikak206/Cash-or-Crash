"""
Train several credit-default models, compare them fairly, and save the winner.

Run from the project root:
    python -m src.train_model              # default: probabilities stay "honest"
    python -m src.train_model --balanced   # use class weights for the 22% imbalance

How the comparison works
------------------------
1. Every model is scored with 5-fold stratified cross-validation on the TRAIN set only.
2. The winner is chosen by mean cross-validated ROC-AUC (never by the test set).
3. Every model is then refitted on the full train set and scored once on the test set,
   so the results table can be quoted in the report.
"""
import argparse
import json
import time
import warnings
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_preprocessing import load_processed_data

warnings.filterwarnings("ignore")


def build_models(balanced: bool = False, neg_pos_ratio: float = 1.0) -> dict:
    """Return the candidate models. XGBoost / LightGBM are skipped if not installed."""
    cw = "balanced" if balanced else None
    models = {
        # Scaler first: logistic regression needs features on a similar scale
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight=cw)),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=30,
            class_weight="balanced_subsample" if balanced else None,
            n_jobs=-1, random_state=config.RANDOM_STATE),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg_pos_ratio if balanced else 1.0,
            eval_metric="auc", n_jobs=-1, random_state=config.RANDOM_STATE)
    except ImportError:
        print("! xgboost not installed - skipping it")
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            class_weight=cw, n_jobs=-1, random_state=config.RANDOM_STATE, verbose=-1)
    except ImportError:
        print("! lightgbm not installed - skipping it")
    return models


def make_ensemble(models: dict):
    """Soft-voting ensemble = average the probabilities of the tree models."""
    members = [(n, models[n]) for n in ("xgboost", "lightgbm", "random_forest") if n in models]
    if len(members) < 2:
        return None
    return VotingClassifier(estimators=members, voting="soft")


def score_model(name, model, X_train, y_train, X_test, y_test, cv):
    """Cross-validate on train, refit on full train, score once on test."""
    start = time.time()
    cvs = cross_validate(model, X_train, y_train, cv=cv, n_jobs=1,
                         scoring={"auc": "roc_auc", "pr_auc": "average_precision"})
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    row = {
        "model": name,
        "cv_auc_mean": cvs["test_auc"].mean(),
        "cv_auc_std": cvs["test_auc"].std(),
        "cv_pr_auc_mean": cvs["test_pr_auc"].mean(),
        "test_auc": roc_auc_score(y_test, proba),
        "test_pr_auc": average_precision_score(y_test, proba),
        "seconds": time.time() - start,
    }
    return row


def run_experiment(models, X_train, y_train, X_test, y_test):
    """Score all models (+ ensemble). Returns (results_df, fitted_models_dict)."""
    cv = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    rows, fitted = [], {}

    def attempt(name, model):
        print(f"Training {name} ...", flush=True)
        try:
            rows.append(score_model(name, model, X_train, y_train, X_test, y_test, cv))
            fitted[name] = model
            print(f"   CV AUC = {rows[-1]['cv_auc_mean']:.4f}   ({rows[-1]['seconds']:.0f}s)")
        except Exception as e:                      # one broken model must not stop the rest
            msg = " ".join(str(e).split())[:200]
            print(f"   ! {name} failed and was skipped: {type(e).__name__}: {msg}")

    for name, model in models.items():
        attempt(name, model)

    ensemble = make_ensemble(fitted)
    if ensemble is not None:
        attempt("voting_ensemble", ensemble)

    results = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)
    return results, fitted


def save_outputs(results, fitted, X_train, balanced):
    """Save the winning model, the comparison table and a small info file."""
    best_name = results.loc[0, "model"]
    config.BEST_MODEL_PATH.parent.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)

    joblib.dump(fitted[best_name], config.BEST_MODEL_PATH, compress=3)
    size_mb = config.BEST_MODEL_PATH.stat().st_size / 1e6
    results.round(4).to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)

    info = {
        "best_model": best_name,
        "selected_by": f"mean {config.N_FOLDS}-fold CV ROC-AUC on the training set",
        "class_weights_used": balanced,
        "cv_auc_mean": round(float(results.loc[0, "cv_auc_mean"]), 4),
        "test_auc": round(float(results.loc[0, "test_auc"]), 4),
        "test_pr_auc": round(float(results.loc[0, "test_pr_auc"]), 4),
        "n_train_rows": int(len(X_train)),
        "features": list(X_train.columns),
        "random_state": config.RANDOM_STATE,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    (config.RESULTS_DIR / "best_model_info.json").write_text(json.dumps(info, indent=2))

    print(f"\nBest model: {best_name}  (saved to {config.BEST_MODEL_PATH.name}, {size_mb:.1f} MB)")
    if size_mb > 50:
        print("! Model file is large; GitHub rejects files over 100 MB.")
    return best_name


def main():
    parser = argparse.ArgumentParser(description="Train and compare credit default models")
    parser.add_argument("--balanced", action="store_true",
                        help="use class weights to handle the 22%% default imbalance")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_processed_data()
    print(f"Train {X_train.shape}, test {X_test.shape}, default rate {y_train.mean():.3f}")
    neg_pos = float((y_train == 0).sum() / (y_train == 1).sum())

    models = build_models(balanced=args.balanced, neg_pos_ratio=neg_pos)
    results, fitted = run_experiment(models, X_train, y_train, X_test, y_test)
    if results.empty:
        raise SystemExit("No model trained successfully.")

    print("\n=== Model comparison (sorted by CV AUC) ===")
    print(results.round(4).to_string(index=False))
    save_outputs(results, fitted, X_train, args.balanced)


if __name__ == "__main__":
    main()
