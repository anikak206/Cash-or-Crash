# Analytical Model Validation Document

This document explains how the model was checked, what the checks show, and where the checks stop.
All figures come from `results/model_comparison.csv`, `results/metrics.json`,
`results/threshold_table.csv` and `results/risk_band_summary.csv`.

## 1. Validation design

| Step | What was done | Why |
|---|---|---|
| Split | One stratified 80/20 split with seed 42: 24,000 training and 6,000 test customers, the same 22.1% default rate in both | Keeps the test set a fair stand-in for new customers |
| Test set kept apart | The test set is not used for training, scaling, filling missing values or choosing the model | Prevents leakage (risk R1) |
| Missing values | Filled with medians calculated on the training part only | No information flows from test to training |
| Scaling | Logistic regression is wrapped in a pipeline with a scaler, so the scaler is fitted again inside every cross-validation fold | No leakage between folds |
| Model selection | 5-fold stratified cross-validation (shuffled, seed 42) on the training set; the model with the highest mean ROC-AUC wins | The choice never uses test data |
| Final report | Every model is refitted on the full training set and scored once on the test set | Gives numbers that can be quoted |
| Class imbalance | No class weights in the reported run; judged with ROC-AUC, PR-AUC, KS and Brier | Keeps predicted probabilities honest for the risk bands |

## 2. Candidate models and settings

| Model | Settings | Notes |
|---|---|---|
| Logistic regression | Scaled inputs, up to 1,000 iterations | Simple baseline |
| Random forest | 200 trees, depth at most 12, at least 30 customers per leaf | Averages many decision trees |
| XGBoost | 300 trees, depth 4, learning rate 0.05, 80% of rows and columns per tree | Gradient boosting |
| LightGBM | 300 trees, 31 leaves, learning rate 0.05, 80% of rows and columns per tree | Gradient boosting |
| Voting ensemble | Soft vote: the average probability of XGBoost, LightGBM and random forest | Chosen model |

These settings are sensible defaults. **No hyper-parameter search was run**, so every model could still improve a little.

## 3. Results

| Model | CV ROC-AUC (mean) | CV spread (std) | CV PR-AUC | Test ROC-AUC | Test PR-AUC | Time, seconds |
|---|---|---|---|---|---|---|
| **Voting ensemble** | **0.7862** | 0.0049 | 0.5644 | **0.7808** | 0.5655 | 35.9 |
| XGBoost | 0.7855 | 0.0057 | 0.5620 | 0.7802 | 0.5634 | 9.3 |
| Random forest | 0.7854 | 0.0045 | 0.5594 | 0.7783 | 0.5588 | 11.7 |
| LightGBM | 0.7803 | 0.0044 | 0.5568 | 0.7774 | 0.5604 | 7.6 |
| Logistic regression | 0.7424 | 0.0106 | 0.5079 | 0.7245 | 0.4974 | 0.8 |

Time covers cross-validation plus the final fit.

## 4. Findings

1. **Tree models clearly beat logistic regression.** The cross-validated gap is about 0.044 ROC-AUC, several times the fold-to-fold spread (0.004 to 0.011), and the test scores agree (0.7808 against 0.7245). The relationship between payment history and default is not a straight line, and trees capture it.
2. **The top three models are practically tied.** The ensemble leads XGBoost by 0.0007 and random forest by 0.0008 in mean CV ROC-AUC, far less than the spread of about 0.005 between folds. They should be treated as equally accurate. The ensemble was chosen because the rule "highest mean CV ROC-AUC" was fixed beforehand, and it costs about 4 times the training time of XGBoost (35.9 s against 9.3 s). A single XGBoost model would be a simpler alternative in production.
3. **No sign of overfitting in the tree models.** Cross-validation and test scores differ by at most 0.007 (ensemble 0.0054, XGBoost 0.0053, random forest 0.0071, LightGBM 0.0029).
4. **Logistic regression drops more on the test set (0.018).** This is larger than its own fold spread (0.011). The cause has not been investigated.
5. **The test result is reliable to within about 0.016.** The 95% interval for the ensemble's test ROC-AUC is roughly 0.765 to 0.796 (Hanley-McNeil approximation), above the 0.75 target.
6. **The risk bands rank customers correctly.** Actual default rates are 12.7% (Low), 41.6% (Medium) and 70.4% (High), against 22.1% overall. The cut-offs 0.30 and 0.60 were set before the final model was trained and were not tuned on the test set.
7. **Cut-off trade-off.** At 0.30, 23.1% of customers are flagged, 55.8% of defaulters are caught and 53.4% of flagged customers really default. Raising the cut-off to 0.60 flags 9.5%, catches 30.1% and raises precision to 70.4%.

## 5. Validation checklist

| Check | Method | Result | Evidence |
|---|---|---|---|
| No test leakage | Code review of the split, imputation, scaling and selection steps; automated test of the split | Passed | `src/data_preprocessing.py`, `src/train_model.py`, `tests/test_preprocessing.py` |
| No overfitting | Compare cross-validation and test scores | Gap 0.003 to 0.007 for tree models | `results/model_comparison.csv` |
| Stable across folds | Spread of the 5 fold scores | 0.0044 to 0.0057 for tree models | `results/model_comparison.csv` |
| Result is not luck | Confidence interval for the test ROC-AUC | About 0.765 to 0.796 | calculated from `results/metrics.json` |
| Beats a trivial answer | Compare with guessing the default rate | PR-AUC 2.6 times higher; Brier 22.1% lower | `results/metrics.json` |
| Business meaning | Default rate by risk band | Rises 12.7%, 41.6%, 70.4% | `results/risk_band_summary.csv` |
| Code behaves | Automated tests | 23 passed | `tests/` |
| Requirements | Compare results with the targets | NFR-01 to NFR-04 met | `docs/13_metrics.md` |

## 6. Limits of this validation

| Limit | Effect | Related risk |
|---|---|---|
| Random split, not a split by date | The score may be optimistic for future customers | R5 |
| One dataset from one bank in 2005 | Results may not carry over to other banks or years | R4 |
| No hyper-parameter search | Scores could improve slightly | none |
| Calibration not scored numerically | Band cut-offs rest on the Brier score and the saved calibration plot only | R6 |
| No results split by group | Unfair outcomes for any group would not show up here | R3 |
| One test set | Confidence interval is an approximation, not a repeated experiment | none |

## 7. Next validation steps

1. Tune settings with an inner cross-validation, keeping the test set untouched.
2. Test on a later time period if dates become available.
3. Add a numeric calibration check and, if needed, recalibrate the probabilities.
4. Report error and approval rates by sex, age and marital status, and compare with a model that leaves these inputs out.
5. Bootstrap the test set for confidence intervals on every metric.

## 8. Conclusion

Within these limits, the model is validated as meeting requirements NFR-01 to NFR-04: it ranks
defaulters well (ROC-AUC 0.7808, KS 0.427), its probabilities beat guessing (Brier 0.1342 against
0.1723), and its risk bands separate customers sensibly. It is a sound basis for a pilot, not
yet evidence for production use.
