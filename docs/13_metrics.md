# Defined Performance Metrics List

All results below are measured on the **6,000 held-out test customers** (1,327 defaulters and
4,673 non-defaulters, a default rate of 22.12%). The model is the soft-voting ensemble chosen by
`src/train_model.py`. Numbers come from `results/metrics.json`, `results/threshold_table.csv`,
`results/risk_band_summary.csv` and `results/model_comparison.csv`.

## 1. Metrics, meaning and results

| Metric | Plain meaning | Calculation | Scale | Result | Target | Verdict |
|---|---|---|---|---|---|---|
| ROC-AUC | Chance that a randomly chosen defaulter gets a higher score than a randomly chosen non-defaulter | Area under the ROC curve | 0.5 = random guessing, 1.0 = perfect | **0.7808** | 0.75 or higher (NFR-01) | Met |
| PR-AUC | How well defaulters are found while keeping the flagged group precise; sensitive to how rare defaulters are | Area under the precision-recall curve (average precision) | Guessing scores the default rate, 0.2212; 1.0 = perfect | **0.5655**, which is 2.6 times the guessing level | None set | Reported |
| Gini | ROC-AUC rescaled the way credit scoring reports it | 2 x ROC-AUC - 1 | 0 = random, 1 = perfect | **0.5616** | None set | Reported |
| KS statistic | Largest gap between the share of defaulters and the share of non-defaulters caught as the cut-off moves | Maximum of (true positive rate - false positive rate) | 0 to 1, higher is better | **0.4270** | 0.35 or higher (NFR-02) | Met |
| Brier score | Average squared error of the predicted probabilities (lower is better) | Mean of (probability - actual outcome) squared | 0 = perfect; always guessing 22.12% scores 0.1723 | **0.1342**, which is 22.1% lower than guessing | Below 0.172 (NFR-03) | Met |
| Cross-validated ROC-AUC | ROC-AUC averaged over 5 folds of the training data; used to choose the model | Mean of 5 fold scores | as ROC-AUC | **0.7862** (spread 0.0049) | Used for selection | Reported |

**Uncertainty:** with 1,327 defaulters and 4,673 non-defaulters the test ROC-AUC has a standard error of about 0.008
(Hanley-McNeil approximation). A 95% interval is roughly **0.765 to 0.796**, so even the low end stays above the 0.75 target.

## 2. Precision, recall and F1 at different cut-offs

A customer is "flagged" when the predicted default probability is at or above the cut-off.

| Cut-off | Customers flagged | Precision | Recall | F1 | Accuracy (for comparison) |
|---|---|---|---|---|---|
| 0.20 | 34.7% | 0.427 | 0.669 | 0.521 | 72.8% |
| 0.25 | 27.6% | 0.486 | 0.607 | 0.539 | 77.1% |
| **0.30** (Low / Medium boundary) | 23.1% | 0.534 | 0.558 | 0.546 | 79.5% |
| 0.40 | 16.4% | 0.610 | 0.453 | 0.520 | 81.5% |
| 0.50 | 11.8% | 0.661 | 0.353 | 0.460 | 81.7% |
| **0.60** (Medium / High boundary) | 9.5% | 0.704 | 0.301 | 0.422 | 81.8% |

- **Precision:** of the flagged customers, the share who really defaulted.
- **Recall:** of all real defaulters, the share that were flagged.
- **F1:** a single score that balances precision and recall.
- **Why accuracy is not the main metric:** flagging nobody at all is already 77.88% accurate. At a 0.50 cut-off the model reaches 81.7%, only 3.8 points better, while catching just 35.3% of defaulters.
  A lower cut-off finds more defaulters but also flags more good customers. The right cut-off depends on the bank's cost of each mistake (see risk R7).

## 3. Risk-band metrics (NFR-04)

| Band | Customers | Share of customers | Actual default rate | Times the overall rate | Defaulters inside |
|---|---|---|---|---|---|
| Low (below 0.30) | 4,614 | 76.9% | 12.7% | 0.57 | 587 |
| Medium (0.30 to below 0.60) | 818 | 13.6% | 41.6% | 1.88 | 340 |
| High (0.60 and above) | 568 | 9.5% | 70.4% | 3.18 | 400 |

- The actual default rate rises from 12.7% to 41.6% to 70.4%, so the bands rank risk in the right order.
- The High band defaults at 3.2 times the overall rate, which clears the "at least 2 times" target.
- The 23.1% of customers outside the Low band contain 740 of the 1,327 defaulters (55.8%).

## 4. Check against the requirement sheet

| Requirement | Target | Result | Verdict |
|---|---|---|---|
| NFR-01 | ROC-AUC 0.75 or higher | 0.7808 | Met |
| NFR-02 | KS 0.35 or higher | 0.4270 | Met |
| NFR-03 | Brier below 0.172 | 0.1342 | Met |
| NFR-04 | Bands rise Low to Medium to High; High at least 2 times the overall rate | 12.7%, 41.6%, 70.4%; High is 3.18 times | Met |
| NFR-09 | All automated tests pass | 23 passed | Met |
| NFR-05, NFR-06 | Prediction speed | Not yet measured | Pending (simulation report) |

## 5. Not measured yet

- **Calibration:** the curve is saved in `results/plots/calibration_curve.png`. No numeric target has been set (risk R6).
- **Fairness by group:** error and approval rates by sex, age and marital status (risk R3).
- **Stability over time:** the split is random, not by date (risk R5).
