# Literature and Reference Log

## 1. How these references were chosen

This is a focused reading list, not a systematic review. It covers five questions the project had to answer:
which data and earlier work to compare with, which models to use, how to handle class imbalance and judge
probability models, how to explain predictions, and how to think about fairness.

**Checked column:** "Checked" means the details were confirmed against a publisher, repository or index page while preparing this log.
"Standard" means a well-known reference whose details come from memory: confirm the volume, pages and DOI in Google Scholar
or the publisher's page before you submit.

## 2. Reference log

### A. Data and earlier credit-scoring work

| # | Reference | What it shows | How this project uses it | Limit, or how this project differs | Checked |
|---|---|---|---|---|---|
| 1 | Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473-2480. | Source of the dataset (30,000 Taiwan clients). Compares six data mining methods and proposes the "Sorting Smoothing Method" to estimate the real probability of default; a neural network was the best at estimating it | Dataset, and the idea that the probability matters more than a yes/no label | Different evaluation method, so scores are not directly comparable. This project adds gradient boosting, an ensemble, SHAP and an API | Checked |
| 2 | Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: a review. *Journal of the Royal Statistical Society, Series A*, 160(3), 523-541. | Reviews the statistical methods used to build consumer credit scoring models | Background for using logistic regression as the baseline | Predates modern boosting and explanation tools | Standard |
| 3 | Baesens, B., Van Gestel, T., Viaene, S., Stepanova, M., Suykens, J., & Vanthienen, J. (2003). Benchmarking state-of-the-art classification algorithms for credit scoring. *Journal of the Operational Research Society*, 54(6), 627-635. | Benchmark on eight real credit-scoring data sets. Found that simple classifiers such as logistic regression and linear discriminant analysis also perform very well | Justifies keeping logistic regression as a baseline | Older methods only | Checked |
| 4 | Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research*, 247(1), 124-136. https://doi.org/10.1016/j.ejor.2015.05.030 | Large benchmark of 41 classifiers on eight credit-scoring data sets; introduces ensemble selection to credit scoring and looks at the financial impact of scorecards | Supports comparing several model families and using an ensemble | Different data sets and metrics; no explanation or deployment | Checked |

### B. Models and tools

| # | Reference | What it shows | How this project uses it | Limit, or how this project differs | Checked |
|---|---|---|---|---|---|
| 5 | Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. | Many randomised trees, averaged, give a strong and stable classifier | Random forest model | Large forests can be big files; settings here limit depth and leaf size | Standard |
| 6 | Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232. | Introduces gradient boosting: add small trees that fix the remaining errors | Principle behind XGBoost and LightGBM | Theory paper; the libraries add engineering | Standard |
| 7 | Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. | A fast, regularised implementation of gradient boosting | XGBoost model | Probabilities may need calibration (see 13) | Standard |
| 8 | Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*, 3146-3154. | An efficient boosting implementation that scales to large data | LightGBM model | Same as above | Standard |
| 9 | Dietterich, T. G. (2000). Ensemble methods in machine learning. In *Multiple Classifier Systems (MCS 2000)*, Lecture Notes in Computer Science 1857, 1-15. Springer. | Explains why combining several models often beats a single one | Soft-voting ensemble | Gains depend on the models making different errors; here the three are close | Standard |
| 10 | Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830. | The scikit-learn library | Pipelines, cross-validation, metrics, logistic regression, random forest | A software reference | Standard |

### C. Imbalance, metrics and probabilities

| # | Reference | What it shows | How this project uses it | Limit, or how this project differs | Checked |
|---|---|---|---|---|---|
| 11 | Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357. | Balances classes by creating synthetic examples of the rare class | Used in the baseline notebook only | Changes predicted probabilities, so the final models do not use it | Standard |
| 12 | Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432. | On imbalanced data the precision-recall view reveals differences that the ROC curve hides | Reports PR-AUC next to ROC-AUC | Simulation-based argument | Standard |
| 13 | Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625-632. | Some learners, such as boosted trees, give distorted probabilities that calibration methods can correct | Reason to check calibration and the Brier score (risk R6) | Calibration step not yet added to the project | Standard |
| 14 | Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*, 143(1), 29-36. | Meaning of AUC and a formula for its standard error | Confidence interval for the test ROC-AUC | An approximation | Standard |
| 15 | Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1-3. | The Brier score for probability forecasts | Brier score in `results/metrics.json` | Originally for weather forecasts | Standard |

### D. Explainability and fairness

| # | Reference | What it shows | How this project uses it | Limit, or how this project differs | Checked |
|---|---|---|---|---|---|
| 16 | Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*, 4765-4774. | Introduces SHAP: one importance value per feature for a prediction, based on Shapley values | `src/explain.py` | Shows what the model used, not what caused the outcome; estimates are slow and approximate | Checked |
| 17 | Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning: Limitations and Opportunities*. fairmlbook.org. | Textbook on defining and testing fairness in decision systems | Basis for the fairness risk (R3) and the planned group-wise checks | Not yet applied to this model | Standard |

## 3. What the reading means for this project

1. **Same data, new angle.** Yeh and Lien stress the quality of the predicted probability. This project does the same (Brier score, calibration plot, risk bands) and adds modern tree models, an ensemble, explanations and a service.
2. **Simple models stay in.** Baesens et al. and Lessmann et al. show that logistic regression is a strong baseline while stronger methods and ensembles add accuracy. The results agree: the tree models lead logistic regression by about 0.04 ROC-AUC.
3. **Probabilities need care.** Niculescu-Mizil and Caruana warn that boosted models can give distorted probabilities. A numeric calibration check is on the list of next validation steps.
4. **Imbalance is handled by measurement.** Following Saito and Rehmsmeier, PR-AUC is reported next to ROC-AUC, and class weighting is left off so the probabilities stay interpretable.
5. **Explanation is required.** SHAP gives per-customer reasons; fairness literature (Barocas et al.) motivates the planned checks by sex, age and marital status.

## 4. Gaps this project does not close

- No comparison with the neural network results of Yeh and Lien on the same footing.
- No time-based validation, no calibration step, no group-wise fairness test.
- Contribution is an engineering case study that combines established methods, not a new algorithm.
