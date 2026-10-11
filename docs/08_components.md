# Component Identification and Characterization Sheet

## 1. Components

| ID | Component | What it does | Input | Output | Technology | File | Key settings |
|---|---|---|---|---|---|---|---|
| C1 | Raw data store | Holds the original customer records | none | 30,000 rows x 25 columns | CSV, pandas | `data/raw/credit_data.csv` | UCI Taiwan credit card data |
| C2 | Cleaning | Drops `ID`, renames the target, groups unknown category codes | raw table | clean table (24 columns) | pandas | `src/data_preprocessing.py` | EDUCATION 0, 5, 6 become 4; MARRIAGE 0 becomes 3 |
| C3 | Feature engineering | Adds 6 behaviour features and returns a new table | clean table | 29 inputs + target | pandas, numpy | `src/feature_engineering.py` | clip at 0 in the payment ratio; limit of 0 gives NaN |
| C4 | Splitter | One stratified split; fills gaps using training medians; saves four CSV files | table with features | `X_train`, `X_test`, `y_train`, `y_test` | scikit-learn | `src/data_preprocessing.py` | 80/20, seed 42 |
| C5 | Trainer and selector | Trains 5 models, scores each with 5-fold cross-validation, picks the best, refits and saves it | training files | best model and results tables | scikit-learn, XGBoost, LightGBM | `src/train_model.py` | no class weights by default; `--balanced` option |
| C6 | Model files | Stored trained models | none | `best_model.joblib` (7.04 MB, voting ensemble) and `credit_model.joblib` (baseline logistic regression, under 0.01 MB) | joblib | `models/` | load time 0.845 s |
| C7 | Evaluator | Scores the saved model on the test set; writes metrics, tables and plots | model, test files | `metrics.json`, `threshold_table.csv`, `risk_band_summary.csv`, 5 plots | scikit-learn, matplotlib | `src/evaluate.py` | cut-offs 0.20 to 0.60 |
| C8 | Explainer | Computes SHAP values for the model's probability | model, customers | importance table, 4 plots, top 5 reasons per customer | SHAP | `src/explain.py` | 50 reference and 200 explained customers |
| C9 | Predictor | Turns raw customer fields into a probability and a risk band | 23 raw fields | probability, risk level, decision, optional reasons | pandas, joblib | `src/predict.py` | rejects missing fields and a credit limit of 0 |
| C10 | API service | Web service with input checking | JSON customer or batch | JSON result | FastAPI, Pydantic, Uvicorn | `app/app.py` | batch up to 1,000 customers |
| C11 | Configuration | One place for paths, seed, split size and risk cut-offs | none | constants and `risk_band()` | Python | `src/config.py` | cut-offs 0.30 and 0.60 |
| C12 | Synthetic data generator | Makes made-up customers with reliable, stretched or distressed behaviour | count, seed | CSV of customers | numpy, pandas | `src/generate_synthetic.py` | default 500 customers, seed 42 |
| C13 | Benchmark | Measures speed, size and software versions | saved model | `results/benchmark.json` | Python standard library, numpy | `src/benchmark.py` | 100 single calls, batches of 1 to 1,000 |
| C14 | Notebooks | Exploration and the first baseline | data files | charts and tables inside the notebooks | Jupyter | `notebooks/01_eda.ipynb` to `05_explainability.ipynb` | baseline uses SMOTE |
| C15 | Automated tests | Check the cleaning, features, split, prediction, risk bands and API | code | pass or fail | pytest | `tests/` | 23 tests: 2 config, 7 preprocessing, 7 prediction, 7 API |

## 2. Software environment

Recorded by `src/benchmark.py` in `results/benchmark.json`.

| Item | Version |
|---|---|
| Python | 3.14.3 |
| Operating system | Windows 11 (10.0.26200) |
| Processor | Intel64 Family 6 Model 126 Stepping 5, GenuineIntel, 8 logical processors |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| xgboost | 3.4.1 |
| lightgbm | 4.7.0 |
| shap | 0.52.0 |
| joblib | 1.6.0 |
| fastapi | 0.142.2 |
| pydantic | 2.13.5 |
| starlette | 1.7.0 |
| uvicorn | 0.54.0 |
| httpx | 0.28.1 |
| pytest | 9.1.1 |
| matplotlib | 3.11.2 |

`requirements.txt` does not fix these versions yet (risk R14), so another computer may install different ones.

## 3. Interfaces (API)

| Method and path | Purpose | Input | Success answer | Errors |
|---|---|---|---|---|
| `GET /` | Service name and links | none | JSON with `docs` and `health` links | none |
| `GET /health` | Is the service up, and which model is loaded | none | `status`, `model`, `risk_thresholds` | 503 if the model file is missing |
| `POST /predict` | Score one customer | 23 fields; optional `explain=true` | `default_probability`, `risk_level`, `decision`, and `top_reasons` if asked | 422 for a missing or invalid field; 503 if the model file is missing |
| `POST /predict/batch` | Score many customers | `customers`: 1 to 1,000 records | `count` and `predictions` | 422 for an empty or invalid list; 503 if the model file is missing |

Interactive documentation is generated automatically at `/docs`.

## 4. Measured characteristics

| Measure | Value |
|---|---|
| Model file size | 7.04 MB |
| Model load time | 0.845 s |
| Training data in memory (24,000 x 29) | 5.57 MB |
| Feature preparation, one customer (median) | 15.6 ms |
| Model call, one customer (median) | 121.9 ms |
| Whole prediction, one customer (median, 95th percentile) | 111.7 ms, 152.6 ms |
| API request in the same process (median, 95th percentile) | 147.1 ms, 233.5 ms |
| Explanation, one customer (median) | 810 ms |
| Batch of 1, 10, 100, 1,000 customers | 0.1445 s, 0.1270 s, 0.1388 s, 0.1606 s |
| Training time including 5-fold cross-validation | ensemble 35.9 s, XGBoost 9.3 s, random forest 11.7 s, LightGBM 7.6 s, logistic regression 0.8 s |

The model call and the whole prediction were timed in separate runs, so their medians differ slightly.
Requirements NFR-05 (one prediction under 1 second) and NFR-06 (explanation time measured) are met.

## 5. Which features matter most

Average absolute SHAP value, in probability points, over 200 test customers (`results/shap/feature_importance.csv`):

| Rank | Feature | Type | Average effect on the probability |
|---|---|---|---|
| 1 | `max_repayment_delay` | engineered | 0.0476 |
| 2 | `PAY_0` | raw | 0.0421 |
| 3 | `avg_repayment_delay` | engineered | 0.0193 |
| 4 | `credit_utilization` | engineered | 0.0187 |
| 5 | `LIMIT_BAL` | raw | 0.0169 |
| 6 | `BILL_AMT1` | raw | 0.0151 |
| 7 | `payment_to_bill_ratio` | engineered | 0.0103 |
| 8 | `PAY_AMT1` | raw | 0.0100 |
| 9 | `PAY_AMT2` | raw | 0.0092 |
| 10 | `PAY_AMT3` | raw | 0.0086 |

- The two features about repayment delay make up 45% of the total effect shown for these ten.
- Four of the ten are features built in this project, which supports the feature engineering step.
- None of `SEX`, `EDUCATION`, `MARRIAGE` or `AGE` is in the top ten. This does not prove fairness (risk R3 stays open).
