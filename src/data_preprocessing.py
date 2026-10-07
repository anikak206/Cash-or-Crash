"""
Data preprocessing: raw CSV  ->  clean, feature-rich, split train/test CSVs.

Run from the project root:
    python -m src.data_preprocessing
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.feature_engineering import add_features


def load_raw_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw credit card CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: fix target name, drop ID, merge undocumented categories."""
    df = df.copy()

    # Some versions of the dataset call the target "default payment next month"
    if config.OLD_TARGET_COL in df.columns:
        df = df.rename(columns={config.OLD_TARGET_COL: config.TARGET_COL})

    # ID is just a row number - it has no predictive value
    df = df.drop(columns=[config.ID_COL], errors="ignore")

    # EDUCATION: only 1-4 are documented; 0, 5, 6 are unknown -> group as "others" (4)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    # MARRIAGE: only 1-3 are documented; 0 is unknown -> group as "others" (3)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def split_data(df: pd.DataFrame):
    """Add engineered features, then make ONE stratified train/test split."""
    df = add_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,                      # keep the same default % in both parts
        random_state=config.RANDOM_STATE,
    )

    # Fill any missing values using the TRAIN medians only (no peeking at test)
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)
    return X_train, X_test, y_train, y_test


def prepare_and_save() -> None:
    """Full pipeline: load -> clean -> features -> split -> save CSVs."""
    df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = split_data(df)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(config.PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(config.PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(config.PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(config.PROCESSED_DIR / "y_test.csv", index=False)

    print(f"Saved to {config.PROCESSED_DIR}")
    print(f"  X_train: {X_train.shape}   X_test: {X_test.shape}")
    print(f"  default rate  train: {y_train.mean():.3f}   test: {y_test.mean():.3f}")


def load_processed_data():
    """Load the saved split. Notebooks and scripts call this instead of re-splitting."""
    d = config.PROCESSED_DIR
    X_train = pd.read_csv(d / "X_train.csv")
    X_test = pd.read_csv(d / "X_test.csv")
    y_train = pd.read_csv(d / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(d / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare_and_save()
