# Feasibility Calculation Sheet

Every line shows the question, the calculation, the inputs, the result and a verdict. Speed numbers were
measured with `python -m src.benchmark` (file `results/benchmark.json`) on a Windows 11 computer with 8 logical
processors and Python 3.14.3. Accuracy numbers come from `results/metrics.json` and `results/threshold_table.csv`.

## 1. Data feasibility

| ID | Question | Calculation | Result | Verdict |
|---|---|---|---|---|
| D1 | Are there enough customers? | Rows in the dataset | 30,000 (24,000 train, 6,000 test) | Enough |
| D2 | Are there enough defaulters for 29 features? | Training defaulters = 6,636 - 1,327 = 5,309; events per feature = 5,309 / 29 | 183 per feature (a common rule of thumb asks for 10 to 20) | Ample |
| D3 | Is the class imbalance manageable? | Non-defaulters : defaulters = 77.9 : 22.1 | About 3.5 : 1 | Moderate; handled by metrics, no resampling needed |
| D4 | Does the data fit in memory? | 24,000 rows x 29 features | 5.57 MB | Trivial for any laptop |
| D5 | Is there missing data to repair? | Missing values in the raw file | 0 | None to repair |

## 2. Compute feasibility

| ID | Question | Calculation | Result | Verdict |
|---|---|---|---|---|
| C1 | How long does training take? | 35.9 + 9.3 + 11.7 + 7.6 + 0.8 seconds for the ensemble, XGBoost, random forest, LightGBM and logistic regression, each with 5-fold cross-validation | 65.3 seconds, about 1.1 minutes | Monthly retraining is cheap |
| C2 | How big is the model? | Size of `models/best_model.joblib` | 7.04 MB | Small; far below GitHub's 100 MB limit |
| C3 | How long does start-up take? | Time to load the model file | 0.845 seconds, once | Fine |
| C4 | Is special hardware needed? | Hardware used | Ordinary computer, processor only (no GPU was used) | Not needed |

## 3. Speed and capacity

| ID | Question | Calculation | Result | Verdict |
|---|---|---|---|---|
| S1 | How fast is one prediction? | Median of 100 calls | 111.7 ms (95th percentile 152.6 ms) | Within the 1-second target (NFR-05) |
| S2 | How fast is one API request? | Median of 50 in-process calls, including input validation | 147.1 ms (95th percentile 233.5 ms) | Within target |
| S3 | How fast is an explanation? | Median of 3 SHAP calls for one customer | 810 ms | Acceptable for an on-demand "why?" |
| S4 | Single-process throughput | 1 / 0.1117 s for a whole prediction, 1 / 0.1471 s for an API request | About 9.0 and 6.8 customers per second | Enough for a pilot |
| S5 | Batch throughput | 1,000 customers in 0.1606 s, so 1,000 / 0.1606 | About 6,200 customers per second | Fast |
| S6 | Re-scoring a whole portfolio | 1,000,000 / 6,227 (extrapolated, assumes time grows in proportion; not measured) | About 161 seconds, 2.7 minutes | Feasible overnight or on demand |
| S7 | Can one process serve a bank? (illustrative) | 100,000 applications per day = 1.157 per second; 1.157 x 0.147 s | 17% of one process's time | Feasible, with room for peaks |

**Where the time goes.** Scoring 1 customer takes 0.1445 s and scoring 1,000 takes 0.1606 s, so almost all of the
single-call time is a fixed cost, not work per customer. Preparing the features takes 15.6 ms, and the model call
about 122 ms when timed on its own (the two timings come from separate runs, so they differ slightly). For real-time use,
batching requests is about 900 times cheaper per customer. The cause of the fixed cost has not been investigated;
ideas to test are single-threaded prediction in the random forest or a single XGBoost model.

## 4. Accuracy feasibility

| Requirement | Target | Achieved | Verdict |
|---|---|---|---|
| NFR-01 ROC-AUC | 0.75 or higher | 0.7808 | Met |
| NFR-02 KS | 0.35 or higher | 0.4270 | Met |
| NFR-03 Brier score | below 0.172 | 0.1342 | Met |
| NFR-04 Risk bands | rise Low to Medium to High; High at least 2 times overall | 12.7%, 41.6%, 70.4%; High is 3.18 times | Met |

## 5. Decision value (economic feasibility)

The bank's real costs are not known, so this sheet uses relative units. Treat "flag" as "reject".
Rejecting a defaulter saves `C_miss`; rejecting a good customer loses `C_fa`.
Let `k = C_miss / C_fa`. Net benefit on the 6,000 test customers, measured in units of one false-alarm cost:

| Cut-off | k = 1 | k = 2 | k = 4 |
|---|---|---|---|
| 0.20 | -304 | 584 | 2,360 |
| 0.25 | -48 | 757 | 2,367 |
| 0.30 (provisional Low / Medium boundary) | 94 | 834 | 2,314 |
| 0.40 | 217 | 818 | 2,020 |
| 0.50 | 228 | 696 | 1,632 |
| 0.60 (provisional Medium / High boundary) | 232 | 632 | 1,432 |
| Reject everyone (1,327 * k - 4,673) | -3,346 | -2,019 | 635 |
| Approve everyone | 0 | 0 | 0 |

**What the table shows**

1. At every k shown, the best model cut-off beats both "approve everyone" and "reject everyone".
2. The best cut-off depends on k: 0.60 (the edge of the grid) when k = 1, 0.30 when k = 2 and 0.25 when k = 4. The cost-based rule (principle note, section 2.9) predicts 0.50, 0.33 and 0.20.
3. The provisional cut-off 0.30 is the best in this grid when k = 2 and only 2.2% below the best when k = 4 (2,314 against 2,367).
4. At 0.30, flagging pays off whenever a missed default costs more than 646 / 740 = 0.87 times a false alarm.
5. At k = 4 and a cut-off of 0.25 the gain is 2,367 / 6,000 = 0.39 false-alarm units per customer.

**Caveats.** The accounting is simplified: the Medium band goes to manual review rather than rejection, review cost is not included, and recoveries after default are ignored.
At a very high k, rejecting everyone starts to look good (it is already positive at k = 4), which only shows that `C_fa` must include the
interest income lost from good customers. The bank must supply the real costs before the cut-offs are fixed (risk R7).

## 6. Resource, ethical and legal feasibility

| Area | Finding |
|---|---|
| Software and cost | Only free, open-source software; no paid services |
| Skills | Python, scikit-learn, gradient boosting and FastAPI; all documented in this repository |
| Fairness | Among the 10 features with the largest average effect, none is `SEX`, `EDUCATION`, `MARRIAGE` or `AGE`. This is encouraging but not proof of fairness, because other features can act as stand-ins. Group-wise checks are still open (risk R3) |
| Regulation | Not assessed. A real lender would need legal review and a human decision step (risk R15) |

## 7. Verdict

| Area | Verdict |
|---|---|
| Data | Feasible |
| Compute | Feasible |
| Speed | Feasible for a pilot; batching recommended for high volume |
| Accuracy | Feasible: all four accuracy requirements are met |
| Economic value | Feasible if a missed default costs more than about 0.9 times a false alarm; needs the bank's real costs |
| Fairness and law | Conditional on the open fairness checks and a legal review |

## 8. How to reproduce
