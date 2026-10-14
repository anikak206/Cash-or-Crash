# Initial Scale-Up Consideration Note

This note asks one question: what would have to change to take the pilot from a laptop demonstration to a service used by a real lender?
It separates what was **measured**, what is **extrapolated** and what is **not yet known**.

## 1. Where the pilot stands today (measured)

| Item | Today | Source |
|---|---|---|
| Deployment | One Uvicorn process on one computer | `app/app.py` |
| One prediction | 111.7 ms median, 152.6 ms at the 95th percentile | `results/benchmark.json` |
| One API request | 147.1 ms median, 233.5 ms at the 95th percentile | `results/benchmark.json` |
| One explanation | 810 ms median | `results/benchmark.json` |
| Batch of 1,000 customers | 0.1606 s, about 6,200 customers per second | `results/benchmark.json` |
| Model file | 7.04 MB, loads in 0.845 s | `results/benchmark.json` |
| Training | 65.3 s for all five models with 5-fold cross-validation on 24,000 customers | `results/model_comparison.csv` |
| Data | 30,000 customers, 5.57 MB for the training matrix | `results/benchmark.json` |

## 2. What changes with size

| Dimension | Pressure at scale | Calculation | Option | Status |
|---|---|---|---|---|
| Real-time requests | 10 million applications a day is 115.7 per second on average | 115.7 / 6.8 per API process = 17 processes, before allowing for peaks | Several worker processes behind a load balancer | Idea, not built |
| Micro-batching | One call carries a large fixed cost (0.1445 s for 1 customer, 0.1388 s for 100) | A batch of 100 takes 1.4 ms per customer, so one process can score about 720 customers per second | Collect requests for a few milliseconds and score them together | Idea, not built |
| Portfolio re-scoring | A bank re-scores all customers each month | 1,000,000 customers at 6,227 per second is about 161 s (extrapolated, not measured) | Nightly or monthly batch job | Idea, not built |
| Training data growth | 100 times more rows (3 million) | Time about 65.3 s x 100 = 1.8 hours; memory 5.57 MB x 100 = 557 MB (rough straight-line estimate, not measured) | Same code on a bigger machine; sample or subset for tuning | Estimate |
| Explanations | 0.81 s each is too slow to run for every customer | 1 / 0.81 s is about 1.2 per process | Explain on request only, or use a faster explainer for the tree models | Idea, not built |
| Model size | More trees or more features | 7.04 MB today | Not a problem until hundreds of MB | No action |
| Fixed cost per call | Cause not found | See `docs/04_feasibility_sheet.md`, section 3 | Test single-threaded prediction or a single XGBoost model | To investigate |

## 3. Deployment options

| Option | What it is | Good for | Cost of change |
|---|---|---|---|
| A. Today | `python -m uvicorn app.app:app --port 8000` | Demonstration, pilot | none |
| B. More workers | `python -m uvicorn app.app:app --workers 4` on one server | Small increase in traffic | small |
| C. Container | The service and the model in one container image | Same behaviour everywhere, easy to move | medium |
| D. Cloud service with automatic scaling | The container on a managed service | Large or uneven traffic | medium to high |
| E. Batch job | `python -m src.predict --csv in.csv --out out.csv` on a schedule | Portfolio reviews, no live traffic | small (already works) |

A first container recipe, **proposed and not yet built or tested**:

Before this works reliably, package versions must be fixed (section 4), because a model saved with one version of the libraries may not load with another.

## 4. Model lifecycle

| Need | Today | Proposal |
|---|---|---|
| Fixed software versions | `requirements.txt` has no versions (risks R13, R14) | Save the exact versions with `python -m pip freeze` into a lock file, starting from the list in `docs/08_components.md` |
| Model versioning | One file, `best_model.joblib`, overwritten on each training | Save each model with its date and `best_model_info.json`; keep the previous one for rollback |
| Retraining | Run by hand | Monthly or when monitoring raises an alert; takes about a minute at the current size |
| Safe changes | One model in use | Run the new model next to the old one on the same customers before switching |
| File safety | joblib files can run code when loaded (risk R13) | Load only files produced by your own pipeline; store them in a protected place |

## 5. Monitoring in production

| What to watch | Why | How |
|---|---|---|
| Input drift | Customers change over time | Population stability index for each important feature: `PSI = sum of (actual share - expected share) * ln(actual share / expected share)`. A common rule of thumb treats a PSI above 0.25 as a major shift |
| Score drift | A shift in scores means a shift in risk or in the data | Share of customers in Low, Medium and High each week, against the training shares (76.9%, 13.6%, 9.5%) |
| Real outcomes | The only true test of accuracy | Actual default rate per band once results are known, against 12.7%, 41.6% and 70.4% |
| Calibration | The bands need honest probabilities (risk R6) | Calibration curve and Brier score on recent outcomes |
| Service health | Users notice failures first | Response time, share of 422 and 503 answers, `/health` check |
| Fairness | Harm can appear at scale (risk R3) | Approval and error rates by sex, age and marital status |

## 6. Security, privacy and governance

- **Access:** add login (an API key or OAuth) and encrypted connections before real customers are scored (risks R11, R12).
- **Abuse:** add rate limiting; the batch size is already capped at 1,000.
- **Logs:** record the time, the model version and the risk level for every decision, but not the customer's personal details.
- **Human decision:** a person confirms every "Reject" before the customer is told (risk R15).
- **Records for audit:** store the model version, the cut-offs and the top reasons with each decision.

## 7. Roadmap

| Stage | Work | Done when |
|---|---|---|
| 0. Pilot (now) | One process, results and documents in the repository | Requirements NFR-01 to NFR-09 met |
| 1. Harden | Lock package versions, build and test the container, add login and logging, run the tests automatically on every change | Container builds and passes the tests |
| 2. Operate | Monitoring from section 5, scheduled retraining, model versions with rollback | An alert fires on a test drift and a rollback works |
| 3. Fairness and review | Group-wise checks, calibration step, cut-offs set from the bank's real costs (risks R3, R6, R7) | Results accepted by the risk manager and compliance |
| 4. Scale out | Several workers or micro-batching, automatic scaling | Load test meets the speed target at the expected volume |

## 8. What is measured, extrapolated or unknown

| Status | Items |
|---|---|
| Measured | Speed of one prediction, one API request, one explanation and a batch; model size and load time; training time; package versions |
| Extrapolated | Re-scoring one million customers; training with 100 times more data; processes needed for 10 million requests a day |
| Not yet known | Behaviour with many requests at the same time; memory used by each process; delay over a real network; accuracy on newer data; cause of the fixed cost per call |
