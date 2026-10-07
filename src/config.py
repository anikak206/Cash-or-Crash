"""
Central settings for the Credit Default Prediction project.

Every file path, random seed and risk threshold lives HERE, so that
notebooks and scripts all use the same values and never disagree.
"""
from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "credit_model.joblib"

# ---------- Data settings ----------
TARGET_COL = "default"
OLD_TARGET_COL = "default payment next month"   # original UCI column name
ID_COL = "ID"                                   # row number, not a real feature
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------- Risk bands (provisional; tuned properly in Phase 2) ----------
LOW_RISK_MAX = 0.30    # probability below this  -> Low risk
HIGH_RISK_MIN = 0.60   # probability at/above this -> High risk


def risk_band(prob: float) -> str:
    """Turn a default probability into a Low / Medium / High risk decision."""
    if prob < LOW_RISK_MAX:
        return "Low Risk - Approve"
    if prob < HIGH_RISK_MIN:
        return "Medium Risk - Manual Review"
    return "High Risk - Reject"
