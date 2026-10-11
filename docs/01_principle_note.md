# Scientific and Technical Principle Note

## 1. The task in one sentence

Learn a function that takes a customer's 29 features (23 raw inputs and 6 engineered ones) and returns
`p`, the probability that the customer misses next month's payment. Then turn `p` into an action.

A probability is more useful than a yes/no label: the bank can approve safe customers, send
borderline cases to a person, and reject very risky ones, and it can change the cut-offs without retraining.

## 2. Principles used

### 2.1 Logistic regression (baseline)

Adds up the features with weights, then squeezes the total into a number between 0 and 1.

The weights are found by minimising the log loss, which punishes confident wrong answers heavily:

- Strength: simple, and each weight shows how a feature pushes the odds.
- Limit: every feature acts in a straight line on the log-odds, so interactions (for example "late payments AND a high bill") are missed.
- Features have very different scales (a credit limit of 200,000 against an age of 30), so they are standardised first.

### 2.2 Decision trees and random forest

A tree asks yes/no questions such as "is `PAY_0` at least 2?" and splits customers into smaller and
smaller groups. A good split makes each group more pure, measured by the Gini impurity:

One deep tree memorises the training data (overfitting). A random forest grows many trees on random
samples of customers and features, then averages them:

Averaging many different trees cancels out their individual mistakes. Here: 200 trees, depth at most 12,
at least 30 customers per leaf.

### 2.3 Gradient boosting (XGBoost and LightGBM)

Trees are built one after another. Each new small tree is fitted to the mistakes the model still makes:

`h_m` is the new tree and `eta` the learning rate (0.05 here), so every tree makes only a small correction.
Shallow trees (depth 4 in XGBoost, 31 leaves in LightGBM) keep each step simple. The two libraries
implement the same idea; they differ in speed and in how the trees are grown.

### 2.4 Soft-voting ensemble

Different models make different mistakes, and averaging their probabilities smooths those mistakes out.
In this project the three models are already close, so the gain is small (see the validation document).

### 2.5 Class imbalance

Only 22.1% of customers default. Common remedies are weighting the rare class or creating synthetic
examples (SMOTE). Both change the predicted probabilities, so a score of 0.30 no longer means "about a 30% chance".
The risk bands need honest probabilities, so the final training leaves the data as it is and judges the
model with measures that cope with imbalance. SMOTE is used only in the baseline notebook (`notebooks/03_model_training.ipynb`).
The option `--balanced` in `src/train_model.py` allows a comparison.

### 2.6 Judging a probability model

- **Calibration:** among customers given about 0.30, roughly 30% should really default.
- **Cross-validation:** split the training data into 5 parts, train on 4, score on the 5th, repeat 5 times and average. The model is chosen on these scores.
- **Leakage:** any use of test data before the final scoring (for scaling, filling gaps or choosing a model) inflates results. The test set is touched once, at the end.

### 2.7 Explaining a prediction with SHAP

SHAP borrows the Shapley value from game theory: the prediction is shared out fairly among the features.

`phi_0` is the average prediction over reference customers, and each `phi_j` is how many probability
points feature `j` added (+) or removed (-) for this customer. The project estimates these values by
permutation on the predicted probability, using 50 reference customers and explaining 200 test customers.
SHAP shows what the model used, not what caused a customer to default.

### 2.8 Feature engineering

The raw columns are six monthly snapshots. Six summaries describe behaviour over the whole period:
average and worst repayment delay, average bill, average payment, payment-to-bill ratio and credit utilisation
(formulas in `docs/data_dictionary.md`).

### 2.9 From probability to decision

Flag a customer when the expected cost of missing a default is larger than the cost of a false alarm:

A cut-off of 0.30 therefore means the bank treats a missed defaulter as about 2.3 times as costly as
wrongly flagging a good customer; a cut-off of 0.20 means 4 times. The project's cut-offs
(0.30 and 0.60) are provisional: the bank sets the real costs.

## 3. Where each principle lives in the project

| Principle | File |
|---|---|
| Cleaning and one stratified split | `src/data_preprocessing.py` |
| Feature engineering | `src/feature_engineering.py` |
| Models, cross-validation, ensemble | `src/train_model.py` |
| Metrics, calibration, risk bands | `src/evaluate.py`, `src/config.py` |
| SHAP explanations | `src/explain.py` |
| Scoring new customers | `src/predict.py`, `app/app.py` |

## 4. Assumptions behind these principles

- Customers are independent of each other.
- The link between payment behaviour and default stays stable over time. If it changes (drift), the model needs retraining (risk R8).
- Every feature is known at the moment of prediction, and none contains the answer itself.
- The training data represents the customers the model will score (risk R4).
