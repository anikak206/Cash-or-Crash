# Data Dictionary

**Source:** UCI Machine Learning Repository, "Default of Credit Card Clients" (Yeh & Lien, 2009).
30,000 credit card customers in Taiwan, April to September 2005. All money amounts are in NT dollars.

**File:** `data/raw/credit_data.csv` has 25 columns: `ID`, 23 input features and 1 target (`default`).

## 1. Raw columns

| Column | Type | Meaning | Values in this dataset |
|---|---|---|---|
| `ID` | integer | Row number | 1 to 30,000. Dropped during cleaning (no predictive value) |
| `LIMIT_BAL` | numeric | Credit limit | 10,000 to 1,000,000 |
| `SEX` | category | 1 = male, 2 = female | 1, 2 |
| `EDUCATION` | category | 1 = graduate school, 2 = university, 3 = high school, 4 = others | 0 to 6. Codes 0, 5 and 6 are undocumented, so they are grouped into 4 (345 rows) |
| `MARRIAGE` | category | 1 = married, 2 = single, 3 = others | 0 to 3. Code 0 is undocumented, so it is grouped into 3 (54 rows) |
| `AGE` | integer | Age in years | 21 to 79 |
| `PAY_0` | ordinal | Repayment status in September 2005 (most recent month) | -2 to 8 |
| `PAY_2` to `PAY_6` | ordinal | Repayment status in August, July, June, May, April 2005 (there is no `PAY_1`) | -2 to 8 |
| `BILL_AMT1` to `BILL_AMT6` | numeric | Bill statement amount, September back to April (1 = most recent) | -339,603 to 1,664,089. Negative means the customer overpaid |
| `PAY_AMT1` to `PAY_AMT6` | numeric | Amount paid, September back to April (1 = most recent) | 0 to 1,684,259 |
| `default` | binary (target) | 1 = defaulted on the next month's payment, 0 = did not | 22.1% of customers are 1 |

**Repayment status values:** -1 = paid on time, 1 to 8 = months of payment delay.
The values -2 and 0 are not in the official description. They are commonly read as
"no credit use that month" (-2) and "revolving credit, minimum payment made" (0).

## 2. Engineered features (`src/feature_engineering.py`)

| Feature | Formula | Why it helps |
|---|---|---|
| `avg_repayment_delay` | mean of the six `PAY_` columns | overall payment behaviour |
| `max_repayment_delay` | maximum of the six `PAY_` columns | worst month |
| `avg_bill_amt` | mean of `BILL_AMT1` to `BILL_AMT6` | typical amount owed |
| `avg_payment_amt` | mean of `PAY_AMT1` to `PAY_AMT6` | typical amount repaid |
| `payment_to_bill_ratio` | `avg_payment_amt / (max(avg_bill_amt, 0) + 1)` | repayment strength; the `+1` and the clip at 0 avoid dividing by zero |
| `credit_utilization` | `avg_bill_amt / LIMIT_BAL` | how much of the limit is used; empty if the limit is 0 |

**Model input:** 23 raw features (without `ID`) + 6 engineered features = **29 features**.

## 3. Risk bands (`src/config.py`)

| Default probability | Risk level | Decision |
|---|---|---|
| below 0.30 | Low | Approve |
| 0.30 to below 0.60 | Medium | Manual review |
| 0.60 and above | High | Reject |

These cut-offs are provisional business rules. `results/risk_band_summary.csv` shows how
well they separate real defaulters on the test set.

## 4. Sensitive attributes

`SEX`, `EDUCATION`, `MARRIAGE` and `AGE` describe the person, not their payment behaviour.
In real lending, using them can be unfair or illegal. The model currently uses them as inputs,
so this is recorded as a fairness risk in the risk register.

## 5. Other data files

| File | What it is |
|---|---|
| `data/processed/X_train.csv`, `y_train.csv` | 24,000 training customers (stratified 80% split) |
| `data/processed/X_test.csv`, `y_test.csv` | 6,000 held-out test customers (20%), never used for training or model selection |
| `data/sample/sample_customers.csv` | 100 real customers (78 non-default, 22 default), a stratified sample for demos |
| `data/sample/synthetic_customers.csv` | 500 made-up customers from `src/generate_synthetic.py`: the 23 input columns plus a `profile` label (reliable, stretched or distressed). Synthetic data is for demos and stress tests only, never for measuring accuracy |

The processed files are created by `python -m src.data_preprocessing`.
