# First-Pass Performance Prediction Document

## 1. Purpose

Before trusting a machine-learning model, ask how far simple, explainable rules can already go. This document
lays out that first-pass estimate as a ladder, from guessing, through one-line rules and a plain logistic regression,
to the final model, and shows what each step adds. It answers one question: is the extra complexity worth it?

**An honest note on timing.** This document was written after the full models were trained. The simple rules have no
fitted settings and were not tuned, so they carry no risk of leakage. Every number can be re-created with
`python -m src.simulate` (file `results/simulation/auc_ladder.csv`) and the test set of 6,000 customers
(1,327 defaulters, 22.1%).

## 2. The ladder

| Step | What it uses | Training needed | ROC-AUC | KS |
|---|---|---|---|---|
| Guess | The same score for everyone | none | 0.5000 | 0.0000 |
| Rule: high credit utilisation | `credit_utilization` | none | 0.5495 | 0.1356 |
| Rule: low credit limit | `LIMIT_BAL` (lower is riskier) | none | 0.5955 | 0.1567 |
| Rule: latest payment delay | `PAY_0` | none | 0.6762 | 0.3553 |
| Rule: worst delay in six months | `max_repayment_delay` | none | 0.6913 | 0.3758 |
| Logistic regression | all 29 features | about 1 second | 0.7245 | 0.3848 |
| **Final model (voting ensemble)** | all 29 features | about 36 seconds with cross-validation | **0.7808** | **0.4270** |

Requirement NFR-01 asks for a ROC-AUC of 0.75 or higher: only the final model meets it.

## 3. What the ladder shows

1. **Payment history is the main signal.** The two delay rules reach 0.68 to 0.69, while rules on credit limit or credit utilisation reach only 0.55 to 0.60. This agrees with the exploratory analysis and with the SHAP ranking, where the two delay features come first.
2. **A one-line rule is a strong baseline.** The "worst delay" rule gets 0.6913 with no training, and its KS of 0.3758 is already 88% of the final model's 0.4270.
3. **Learning from all features helps.** Logistic regression adds 0.033 ROC-AUC over the best rule, and the ensemble adds another 0.056, for a total gain of 0.090 over the best rule.
4. **The last step is not linear.** The jump from logistic regression to the ensemble shows that interactions between features matter, such as a late payment together with a high bill.

## 4. One-line business rules against the model

Each rule flags customers (treated as "reject" or "review"). Precision and recall are on the 6,000 test customers.

| Rule | Customers flagged | Precision | Recall |
|---|---|---|---|
| `PAY_0` is at least 1 (any delay last month) | 22.1% | 0.498 | 0.497 |
| `PAY_0` is at least 2 (two or more months late) | 10.6% | 0.690 | 0.331 |
| Worst delay is at least 2 in six months | 27.6% | 0.451 | 0.563 |
| Worst delay is at least 3 in six months | 4.2% | 0.638 | 0.122 |
| Model, cut-off 0.60 (High band) | 9.5% | 0.704 | 0.301 |
| Model, cut-off 0.50 | 11.8% | 0.661 | 0.353 |
| Model, cut-off 0.30 (Low / Medium boundary) | 23.1% | 0.534 | 0.558 |

Comparing at a similar size of flagged group:

- **About 11% flagged:** the rule `PAY_0` at least 2 (precision 0.690, recall 0.331) and the model at 0.50 (precision 0.661, recall 0.353) are almost equal. At the strict end, a one-line rule does about as well as the model.
- **About 22 to 23% flagged:** the model (0.30) catches 55.8% of defaulters, against 49.7% for the rule `PAY_0` at least 1, with a higher precision (0.534 against 0.498).
- **About 56% recall:** the model needs to flag 23.1% of customers, while the worst-delay rule needs 27.6%, and the model's precision is higher (0.534 against 0.451).

So the model's advantage lies in the broad middle of the range, where a simple rule gives up either precision or recall.

## 5. Value in decision terms

Using the net-benefit idea from `docs/04_feasibility_sheet.md` (net benefit = TP x k - FP, in units of one false alarm, on the 6,000 test customers):

| k (cost of a missed default compared with a false alarm) | Best one-line rule | Model, best cut-off in the grid | Model advantage |
|---|---|---|---|
| 1 | 242 (`PAY_0` at least 2) | 232 (cut-off 0.60) | about 4% lower |
| 2 | 681 (`PAY_0` at least 2) | 834 (cut-off 0.30) | about 22% higher |
| 4 | 2,079 (worst delay at least 2) | 2,367 (cut-off 0.25) | about 14% higher |

If a false alarm costs as much as a missed default (k = 1), a one-line rule does as well as the model. The model's grid stops at a
cut-off of 0.60, so its true best at k = 1 may be higher. When a missed default costs two to four times a false alarm, the model adds about 14% to 22%.

## 6. Is the machine learning worth it?

| Question | Answer |
|---|---|
| Does any simple rule meet the accuracy target? | No. The best rule reaches 0.6913 against a target of 0.75 |
| How much does the model add? | 0.090 ROC-AUC over the best rule and about 14% to 22% more net benefit when a missed default costs 2 to 4 false alarms |
| What does it cost? | About 36 seconds to train, 7.04 MB, about 0.11 seconds per prediction, and explanations that need SHAP |
| What is the risk? | Less transparent than a rule; reduced by the SHAP explanations and the documented validation |

**Recommendation.** Keep the model for scoring and risk bands. Keep `PAY_0` at least 2 as a visible
sanity rule and fallback: it is easy to explain to customers and regulators, and it matches the model's High band almost exactly.

## 7. Limits

- The rules and the model are compared on the same test set, and the net-benefit numbers depend on the unknown cost ratio k.
- The one-line rules were chosen from what the exploratory analysis showed to be strong, then evaluated once. They were not tuned on the test set.
- The data is from one bank in 2005, so the ladder may look different elsewhere (risk R4).

## 8. How to reproduce

The chart `results/simulation/auc_ladder.png` shows the same ladder as bars.
