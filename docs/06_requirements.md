# Functional and Performance Requirement Sheet

Each requirement has an ID, a priority (Must = needed for the project to work, Should = important
but not blocking), and a way to check it. Targets are measured on the 6,000-customer test set
unless stated otherwise.

## 1. Functional requirements

| ID | Requirement | Priority | How it is checked |
|---|---|---|---|
| FR-01 | The system accepts a customer record with the 23 input fields (`LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`, `PAY_0` to `PAY_6` without `PAY_1`, `BILL_AMT1` to `BILL_AMT6`, `PAY_AMT1` to `PAY_AMT6`) | Must | `src/predict.py`, `tests/test_predict.py` |
| FR-02 | The system rejects invalid input (missing field, credit limit of 0 or less, out-of-range value) with a clear error | Must | `tests/test_predict.py`, `tests/test_api.py` (HTTP 422) |
| FR-03 | The system returns a default probability between 0 and 1 | Must | `tests/test_predict.py` |
| FR-04 | The system maps the probability to Low, Medium or High risk using the cut-offs in `src/config.py` (below 0.30, 0.30 to below 0.60, 0.60 and above) | Must | `tests/test_config.py` |
| FR-05 | On request (`explain=true`) the system returns the top 5 reasons for a customer, each with its direction (raises or lowers risk) | Should | `src/explain.py`, manual API test |
| FR-06 | The system scores up to 1,000 customers in one batch request | Should | `tests/test_api.py` |
| FR-07 | A health endpoint reports whether the service is up and which model is loaded | Should | `tests/test_api.py` |
| FR-08 | A command-line tool scores a whole CSV file of customers | Should | `python -m src.predict --csv ...` |
| FR-09 | Each pipeline stage runs with one command: preprocessing, training, evaluation, explanation | Must | commands listed in the README |
| FR-10 | Training compares at least four model types and saves the best one with a record of its scores | Must | `results/model_comparison.csv`, `results/best_model_info.json` |

## 2. Performance and quality requirements

| ID | Requirement | Target | How it is checked |
|---|---|---|---|
| NFR-01 | Ranking quality | ROC-AUC of 0.75 or higher | `results/metrics.json` |
| NFR-02 | Separation of defaulters from non-defaulters | KS statistic of 0.35 or higher | `results/metrics.json` |
| NFR-03 | Probability quality | Brier score below 0.172, which is the score of always predicting the average default rate (0.221 x 0.779) | `results/metrics.json` |
| NFR-04 | Risk bands are meaningful | Actual default rate rises from Low to Medium to High; High is at least 2 times the overall rate | `results/risk_band_summary.csv` |
| NFR-05 | Speed of a single prediction without explanation | Under 1 second on an ordinary laptop (to be measured and reported in the simulation report) | simulation report |
| NFR-06 | Speed of a prediction with explanation | Measured and reported; slower than NFR-05 is acceptable | simulation report |
| NFR-07 | No test-set leakage | The test set is never used for training, scaling, imputation or model selection | `src/data_preprocessing.py`, `src/train_model.py`, `tests/test_preprocessing.py` |
| NFR-08 | Reproducibility | Random seed fixed to 42; the same commands give the same split and the same sample files | `src/config.py` |
| NFR-09 | Automated testing | All automated tests pass | `python -m pytest` |
| NFR-10 | Privacy | The API does not store the customer data it receives; the repository holds no personal identifiers | `app/app.py`, `data/` |
| NFR-11 | Fairness transparency | Sensitive inputs (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`) are listed openly and the fairness risk is recorded | `docs/data_dictionary.md`, risk register |
| NFR-12 | Portability | Runs on Windows or Linux with the packages in `requirements.txt` | `requirements.txt` |

## 3. Traceability to the objectives

| Objective (problem statement) | Requirements |
|---|---|
| O1 Predict default well | NFR-01, NFR-02, NFR-03 |
| O2 Choose the model fairly | FR-10, NFR-07 |
| O3 Segment customers by risk | FR-04, NFR-04 |
| O4 Make decisions explainable | FR-05, NFR-06, NFR-11 |
| O5 Make the model usable | FR-01, FR-02, FR-03, FR-06, FR-07, FR-08, NFR-05 |
| O6 Be reproducible and tested | FR-09, NFR-08, NFR-09 |
