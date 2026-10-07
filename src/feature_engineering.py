import numpy as np
import pandas as pd

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 6 behaviour-based features and return a NEW dataframe
    (the input dataframe is left untouched).
    """
    df = df.copy()

    # 1. Average repayment delay
    df["avg_repayment_delay"] = df[PAY_COLS].mean(axis=1)

    # 2. Worst repayment delay (worst-case behaviour)
    df["max_repayment_delay"] = df[PAY_COLS].max(axis=1)

    # 3. Average bill amount
    df["avg_bill_amt"] = df[BILL_COLS].mean(axis=1)

    # 4. Average payment made
    df["avg_payment_amt"] = df[PAY_AMT_COLS].mean(axis=1)

    # 5. Payment-to-bill ratio (repayment strength).
    #    Bills can be negative (customer overpaid), so clip at 0 before adding 1;
    #    this guarantees the denominator is never 0.
    df["payment_to_bill_ratio"] = df["avg_payment_amt"] / (df["avg_bill_amt"].clip(lower=0) + 1)

    # 6. Credit utilization; a limit of 0 becomes NaN instead of infinity
    df["credit_utilization"] = df["avg_bill_amt"] / df["LIMIT_BAL"].replace(0, np.nan)

    return df
