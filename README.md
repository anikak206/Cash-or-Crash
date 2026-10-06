# Credit Default Prediction System

A machine learning-based risk assessment system for financial institutions to predict loan default probability, segment customers by risk, and provide explainable credit decisions.

---

## Key Features
- **Default Probability Prediction:** Supervised learning models trained to estimate borrower default risk.
- **Risk Segmentation:** Categorizes applicants into Low, Medium, or High risk tiers.
- **Explainable AI:** Uses SHAP (SHapley Additive exPlanations) values to interpret feature impacts on predictions.
- **Production-Ready Pipeline:** End-to-end Machine Learning pipeline served via FastAPI.
- **Model Evaluation:** Optimization guided by ROC-AUC, Precision-Recall, and F1-score metrics.

---

## Repository Structure

```text
Cash-or-Crash/
├── app/                  # FastAPI application scripts
├── data/                 # Datasets (raw & processed)
├── models/               # Trained model artifacts (.pkl / .joblib)
├── notebooks/            # Jupyter notebooks for EDA and model training
├── src/                  # Core source code (preprocessing, pipeline, utils)
├── .gitignore            # Git ignore rules
├── README.md             # Project documentation
└── requirements.txt      # Python package dependencies