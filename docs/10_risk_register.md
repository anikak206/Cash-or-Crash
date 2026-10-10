# Use Case and Stakeholder Definition

## 1. Purpose of the system

A bank (or card issuer) uses this system to estimate how likely a credit card customer is to miss
the next payment, to place the customer in a Low / Medium / High risk band, and to see the main
reasons behind the result. The system **supports** a human decision. It does not replace it.

## 2. Stakeholders

| Stakeholder | Role | What they need from the system | How the project serves them | Main concern |
|---|---|---|---|---|
| Credit officer | Decides on limits and approvals for individual customers | A fast answer and a reason they can repeat to the customer | `/predict` returns probability and risk level; `explain=true` adds the top 5 reasons | Trusting a score they cannot explain |
| Credit risk manager | Owns portfolio risk and the approval policy | Proof that the model separates good and bad customers, and control of the cut-offs | `results/metrics.json`, `results/risk_band_summary.csv`, `results/threshold_table.csv`, cut-offs in `src/config.py` | Wrong cut-offs causing losses or lost business |
| Customer (applicant) | Person whose credit is judged | A fair decision and an explanation | Reasons per customer; sensitive inputs listed openly in `docs/data_dictionary.md` | Being treated unfairly because of sex, age or marital status |
| Compliance officer / auditor | Checks fairness, privacy and documentation | A reproducible, documented process | Fixed seed, one command per stage, tests, risk register | Unexplainable or biased decisions |
| Data scientist (maintainer) | Trains and improves the model | Clean code, tests, a clear comparison of models | `src/` modules, `tests/`, `results/model_comparison.csv` | Silent errors such as test-data leakage |
| IT / operations | Runs the service | A simple, validated API with a health check | FastAPI app, `/health`, input validation, `requirements.txt` | Crashes, bad input, version problems |
| Bank management | Funds and approves the system | Lower credit losses, defensible decisions | Risk bands and metrics in plain terms | Cost and regulatory exposure |

## 3. Use cases

| ID | Use case | Actor | Trigger | Main steps | Result | Requirements |
|---|---|---|---|---|---|---|
| UC-1 | Score one customer | Credit officer | A customer applies or is reviewed | 1. Enter the 23 fields. 2. System validates them. 3. System returns probability and risk level. | Probability, Low / Medium / High, suggested action | FR-01 to FR-04 |
| UC-2 | Explain a decision | Credit officer, customer-facing staff | Customer asks "why?" | 1. Score the customer with `explain=true`. 2. System returns the top 5 factors and whether each raises or lowers risk. | Plain list of reasons | FR-05 |
| UC-3 | Review a whole portfolio | Credit risk manager | Monthly or quarterly review | 1. Upload a batch (API) or a CSV (command line). 2. System scores every customer. 3. Manager looks at the counts per band. | Scored list and risk-band counts | FR-06, FR-08 |
| UC-4 | Check model quality and retrain | Data scientist, risk manager | New data or a scheduled review | 1. Run preprocessing, training and evaluation. 2. Compare scores with the requirement targets. 3. Keep or replace the model. | Updated `results/` files | FR-09, FR-10 |
| UC-5 | Audit the process | Compliance officer, auditor | Internal or external audit | 1. Read the problem statement, requirements, data dictionary and risk register. 2. Re-run the commands. 3. Compare with the saved results. | Reproduced results and a documented trail | NFR-07 to NFR-11 |
| UC-6 | Check service health | IT / operations | Deployment or monitoring | Call `/health` | Status and loaded model name

@'
# Initial Qualitative Risk Register

This is the first-pass list of what could go wrong with the project, how likely and how serious each
risk is, and what has been done about it. It is qualitative: ratings are judgements (Low, Medium,
High), not measured probabilities.

**Status key:** Mitigated = a control exists and is in the repository. Partly = a control exists
but is not enough. Open = nothing done yet. Accepted = we knowingly live with it and say so.

## 1. Risk list

| ID | Risk | Category | Likelihood | Impact | What we did or will do | Status |
|---|---|---|---|---|---|---|
| R1 | Test data leaks into training, so scores look better than they are | Data / validation | Medium | High | Found and fixed a double-split bug in Phase 1. One stratified split is saved once; the model is chosen on cross-validation of the training set only; `tests/test_preprocessing.py` checks the split | Mitigated |
| R2 | Class imbalance (22.1% defaulters) makes accuracy misleading | Data / metrics | High | Medium | Report ROC-AUC, PR-AUC, KS, Gini, Brier and a precision/recall table. Class weights are off by default to keep probabilities honest; `--balanced` is available for comparison | Mitigated |
| R3 | Unfair outcomes: `SEX`, `EDUCATION`, `MARRIAGE` and `AGE` are model inputs, and default rates differ by group (for example men 24.2% and women 20.8% in the data) | Ethics / legal | High | High | Listed openly in the data dictionary. Still to do: compare error and approval rates by group, and test the model without these inputs | Open |
| R4 | Data is one bank in one country in 2005, so results may not carry over | Data | High | Medium | Stated as a limitation in the problem statement. A real deployment would need local data | Accepted |
| R5 | Random split instead of a time-based split gives an optimistic estimate | Validation | Medium | Medium | Stated as a limitation. Future work: train on earlier months and test on later ones | Accepted |
| R6 | Predicted probabilities are poorly calibrated, so the risk bands mislead | Model | Medium | High | Calibration curve and Brier score in `results/`; the bands are checked against real default rates in `results/risk_band_summary.csv`. Re-check after every retrain | Partly |
| R7 | The cut-offs 0.30 and 0.60 are guesses, not based on loss costs | Business | High | Medium | Marked provisional. `results/threshold_table.csv` shows the trade-off. Needs the bank's cost of a missed default versus a lost good customer | Partly |
| R8 | Model quality drops over time as customer behaviour changes (drift) | Operations | High | High | Not built. The scale-up note describes monitoring and scheduled retraining | Open |
| R9 | People read SHAP explanations as proof of cause | Explainability | Medium | Medium | Documentation says the reasons show what the model used, not why a person defaulted. SHAP is computed on a sample of 200 test customers with 50 reference customers | Partly |
| R10 | Overfitting by the tree models | Model | Medium | Medium | Cross-validation, a held-out test set, limited depth and leaf size (random forest depth 12, minimum 30 per leaf; XGBoost depth 4) | Mitigated |
| R11 | Bad or hostile input to the API | Security | Medium | Medium | Field types and ranges are validated (HTTP 422), batch size is capped at 1,000. There is no login and no rate limiting yet | Partly |
| R12 | Customer data exposed through the service | Privacy | Medium | High | The code does not save requests, and the repository holds only the public dataset and made-up customers. A real deployment needs login, encryption in transit and an access log policy | Partly |
| R13 | Loading a model file can run code (joblib uses pickle) and breaks if library versions differ | Security / operations | Low | High | Only load files from this repository. `requirements.txt` is not pinned yet, so versions can drift | Open |
| R14 | Results differ on another machine because packages are not pinned | Reproducibility | Medium | Medium | Random seed is fixed (42). To do: pin exact package versions | Open |
| R15 | Staff treat "Reject" as final and skip human judgement | Process | Medium | High | The Medium band is manual review and "Reject" is a recommendation. A policy that a person confirms every High decision is still needed | Partly |
| R16 | Made-up customers are mistaken for evidence of accuracy | Reporting | Low | Medium | Files and documents label them synthetic and say accuracy is measured on the real test set | Mitigated |
| R17 | Model file becomes too large for GitHub (100 MB limit) | Engineering | Low | Low | Saved with compression and a size warning in `src/train_model.py` | Mitigated |
| R18 | Undocumented codes in the data (EDUCATION 0, 5, 6; MARRIAGE 0; repayment status -2 and 0) are misread | Data quality | Medium | Low | Unknown education and marriage codes are grouped into "others". The meaning of -2 and 0 is a common reading, not an official one | Accepted |

## 2. Rating scale

| Rating | Likelihood | Impact |
|---|---|---|
| Low | Unlikely during the project or a pilot | Small effect, easy to fix |
| Medium | Could happen | Noticeable effect on results or trust |
| High | Likely or already visible | Wrong decisions, unfair treatment, legal or financial harm |

## 3. Risks to deal with first

Open or partly handled risks that are both likely and serious:

1. **R3 Fairness.** Compare results by sex, age and marital status, and test the model without those inputs.
2. **R8 Drift.** Needs a monitoring and retraining plan before any real use.
3. **R13 and R14 Versions.** Pin package versions so results and model files load the same everywhere.
4. **R11 and R12 Security and privacy.** Add login and encryption before the API is used with real customers.
5. **R7 Cut-offs.** Set them with the bank's own costs.

## 4. Review

This register should be reviewed after every retrain and before any pilot. New risks are added at the bottom with the next free ID.
