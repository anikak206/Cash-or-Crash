"""
Generate SYNTHETIC (made-up) customers in the same 23-column format as the real data.

Use it to demo the API, stress-test batch scoring, and show how the model reacts to
different customer behaviour. It is NOT real data and must not be used to measure
accuracy; accuracy is measured on the real held-out test set (src/evaluate.py).

Each customer follows one of three behaviour profiles:
    reliable   - pays on time, low bills compared with the limit
    stretched  - uses most of the limit, pays only part of the bill, occasional delays
    distressed - bills near or over the limit, pays almost nothing, repeated delays

Run from the project root:
    python -m src.generate_synthetic --n 500 --out data/sample/synthetic_customers.csv
"""
import argparse

import numpy as np
import pandas as pd

from src.predict import RAW_FEATURES

PROFILES = ["reliable", "stretched", "distressed"]
DEFAULT_MIX = (0.60, 0.25, 0.15)


def _pay_status(rng, profile):
    """Six monthly repayment statuses, newest first (PAY_0, PAY_2 ... PAY_6)."""
    if profile == "reliable":
        return rng.choice([-2, -1, 0], size=6, p=[0.15, 0.45, 0.40])
    if profile == "stretched":
        return rng.choice([-1, 0, 1, 2], size=6, p=[0.20, 0.50, 0.20, 0.10])
    start = int(rng.integers(1, 4))                       # distressed: delays that persist
    return np.clip(start + rng.integers(-1, 2, size=6), 1, 8)


def _one_customer(rng, profile):
    limit_choices = {"reliable": [50000, 100000, 200000, 300000],
                     "stretched": [30000, 50000, 100000, 150000],
                     "distressed": [10000, 20000, 50000, 80000]}[profile]
    limit = int(rng.choice(limit_choices))

    use = {"reliable": (0.05, 0.40), "stretched": (0.40, 0.85), "distressed": (0.80, 1.15)}[profile]
    paid = {"reliable": (0.60, 1.00), "stretched": (0.10, 0.40), "distressed": (0.00, 0.08)}[profile]

    base_bill = limit * rng.uniform(*use)
    bills = np.round(base_bill * rng.uniform(0.8, 1.2, size=6))          # month 1 = newest
    pays = np.round(np.roll(bills, -1) * rng.uniform(*paid, size=6))     # pay part of the older bill

    row = {"LIMIT_BAL": limit,
           "SEX": int(rng.choice([1, 2], p=[0.40, 0.60])),
           "EDUCATION": int(rng.choice([1, 2, 3, 4], p=[0.35, 0.47, 0.16, 0.02])),
           "MARRIAGE": int(rng.choice([1, 2, 3], p=[0.45, 0.53, 0.02])),
           "AGE": int(np.clip(rng.normal(35, 9), 21, 70))}
    status = _pay_status(rng, profile)
    for name, value in zip(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"], status):
        row[name] = int(value)
    for i in range(6):
        row[f"BILL_AMT{i + 1}"] = int(bills[i])
    for i in range(6):
        row[f"PAY_AMT{i + 1}"] = int(max(pays[i], 0))
    row["profile"] = profile
    return row


def generate(n: int = 500, seed: int = 42, mix=DEFAULT_MIX) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    profiles = rng.choice(PROFILES, size=n, p=list(mix))
    df = pd.DataFrame([_one_customer(rng, p) for p in profiles])
    return df[RAW_FEATURES + ["profile"]]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic customers")
    parser.add_argument("--n", type=int, default=500, help="number of customers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/sample/synthetic_customers.csv")
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} synthetic customers to {args.out}")
    print(df["profile"].value_counts().to_string())


if __name__ == "__main__":
    main()
