import numpy as np

from src import config
from src.data_preprocessing import clean_data, load_raw_data, split_data
from src.feature_engineering import add_features

NEW_FEATURES = ["avg_repayment_delay", "max_repayment_delay", "avg_bill_amt",
                "avg_payment_amt", "payment_to_bill_ratio", "credit_utilization"]


def sample_raw(n=300):
    return load_raw_data().head(n).copy()


def test_clean_data_drops_id_and_groups_unknown_codes():
    raw = sample_raw()
    raw.loc[0, "EDUCATION"] = 5      # undocumented code
    raw.loc[1, "MARRIAGE"] = 0       # undocumented code
    cleaned = clean_data(raw)

    assert "ID" not in cleaned.columns
    assert len(cleaned) == len(raw)
    assert set(cleaned["EDUCATION"]) <= {1, 2, 3, 4}
    assert set(cleaned["MARRIAGE"]) <= {1, 2, 3}
    assert "ID" in raw.columns        # the input was not modified


def test_clean_data_renames_old_target_name():
    raw = sample_raw(20).rename(columns={config.TARGET_COL: config.OLD_TARGET_COL})
    assert config.TARGET_COL in clean_data(raw).columns


def test_add_features_creates_six_finite_columns():
    out = add_features(clean_data(sample_raw()))
    for col in NEW_FEATURES:
        assert col in out.columns
    assert np.isfinite(out[NEW_FEATURES].to_numpy()).all()


def test_add_features_does_not_change_its_input():
    df = clean_data(sample_raw(50))
    before = df.copy()
    add_features(df)
    assert df.equals(before)


def test_zero_credit_limit_gives_nan_not_infinity():
    df = clean_data(sample_raw(5))
    df.loc[df.index[0], "LIMIT_BAL"] = 0
    out = add_features(df)
    assert np.isnan(out.loc[out.index[0], "credit_utilization"])
    assert not np.isinf(out["credit_utilization"]).any()


def test_negative_bills_do_not_break_the_ratio():
    df = clean_data(sample_raw(5))
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    df.loc[df.index[0], bill_cols] = -1          # average bill of exactly -1
    out = add_features(df)
    assert np.isfinite(out.loc[out.index[0], "payment_to_bill_ratio"])


def test_split_is_stratified_disjoint_and_clean():
    X_train, X_test, y_train, y_test = split_data(clean_data(load_raw_data()))

    assert len(X_train) == 24000 and len(X_test) == 6000
    assert X_train.shape[1] == 29
    assert "ID" not in X_train.columns and config.TARGET_COL not in X_train.columns
    assert abs(y_train.mean() - y_test.mean()) < 0.005       # same default rate in both parts
    assert set(X_train.index).isdisjoint(set(X_test.index))  # no customer in both parts
    assert not X_train.isna().any().any() and not X_test.isna().any().any()
