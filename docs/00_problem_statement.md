# Problem Statement and Objectives

**Project title:** Cash or Crash: An Explainable Machine Learning System for Predicting Credit Card Default Probability and Segmenting Customer Risk

## 1. Background

When a bank gives a customer a credit card, it lends money without knowing for sure that the
customer will repay. If many customers miss payments (default), the bank loses money. Banks
therefore need to decide, for each customer, how risky the lending is.

Two things make this hard:

- A simple yes/no answer is not enough. A bank wants a **probability**, so it can approve safe
  customers, send borderline cases to a human, and reject very risky ones.
- Lenders are expected to **explain** their decisions to customers and regulators. A model that
  only says "high risk" without a reason is difficult to trust or defend.

## 2. Problem statement

Given a customer's basic details (credit limit, sex, education, marital status, age) and six
months of repayment status, bill amounts and payment amounts, **predict the probability that the
customer will miss the next month's payment, assign a Low / Medium / High risk level, and show
which factors drove the result.**

## 3. What this project adds

| Gap in a simple solution | What this project does about it |
|---|---|
| One model, chosen without comparison | Compares five models (logistic regression, random forest, XGBoost, LightGBM, soft-voting ensemble) with 5-fold cross-validation and picks the winner on the training data only |
| Accuracy hides the problem (only 22.1% of customers default) | Reports ROC-AUC, PR-AUC, Gini, KS and Brier score, and precision/recall at several cut-offs |
| Test data leaking into training | One stratified 80/20 split, saved once; the test set is never used for training or for choosing the model |
| Only a score, no reason | SHAP explanations: global feature importance and the top reasons for each customer |
| A notebook nobody else can use | A REST API (`/predict`, `/predict/batch`, `/health`), a command-line scorer and automated tests |

## 4. Objectives

| ID | Objective | How it is measured | Target | Evidence in the repository |
|---|---|---|---|---|
| O1 | Predict default well | ROC-AUC on the held-out test set (6,000 customers) | 0.75 or higher | `results/metrics.json` |
| O2 | Choose the model fairly | Number of model families compared with stratified 5-fold CV; selection uses CV scores only | At least 4 | `results/model_comparison.csv`, `results/best_model_info.json` |
| O3 | Segment customers by risk | Actual default rate per risk band on the test set | Low < overall rate < Medium < High, and High at least 2 times the overall rate | `results/risk_band_summary.csv`, `results/plots/risk_bands.png` |
| O4 | Make decisions explainable | SHAP global importance, and top 5 reasons per customer on request | Available for every model we train | `results/shap/`, API option `explain=true` |
| O5 | Make the model usable | REST API with input validation | 3 endpoints; invalid input returns HTTP 422 | `app/app.py`, `tests/test_api.py` |
| O6 | Be reproducible and tested | Fixed random seed, one command per pipeline stage, automated tests | All tests pass | `tests/`, `python -m pytest` |

## 5. Scope

**In scope**
- The UCI "Default of Credit Card Clients" data: 30,000 customers, April to September 2005, 22.1% defaulters.
- Data cleaning, feature engineering, model comparison, evaluation, explanation, prediction API.
- Documents that describe the design, validation and risks.

**Out of scope**
- Connecting to a real bank system or using real customer data.
- Legal or regulatory certification, and a full fairness audit.
- Continuous monitoring of the model after deployment (discussed in the scale-up note only).

## 6. Assumptions and limitations

- The data comes from one bank in one country in one period (2005). Results may not carry over to other banks, countries or years.
- The train/test split is random, not by date, so the model has not been tested on "future" customers.
- `SEX`, `EDUCATION`, `MARRIAGE` and `AGE` are used as inputs. In real lending this can be unfair or illegal. It is recorded in the risk register.
- The Low / Medium / High cut-offs (0.30 and 0.60) are provisional business rules, not legal or regulatory limits.
- Synthetic customers (`data/sample/synthetic_customers.csv`) are for demonstration only. Accuracy is measured on real held-out data.

## 7. Success criteria

The project is successful if objectives O1 to O6 are met and the evidence files above exist in
the repository and agree with each other.
