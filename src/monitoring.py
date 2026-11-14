
import pandas as pd
import numpy as np

import joblib

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "outputs/models"

TABLE_DIR = ROOT / "outputs/tables"

splits = joblib.load(
    MODEL_DIR / "data_splits.joblib"
)

X_train = splits["X_train"]

X_test = splits["X_test"]


# -----------------------------------------
# POPULATION STABILITY INDEX
# -----------------------------------------

def calculate_psi(expected, actual, bins=10):

    expected = pd.Series(expected).dropna()

    actual = pd.Series(actual).dropna()

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)

    breakpoints = np.unique(
        expected.quantile(quantiles).values
    )

    if len(breakpoints) < 3:
        return np.nan

    breakpoints[0] = -np.inf

    breakpoints[-1] = np.inf

    expected_counts = pd.cut(
        expected,
        bins=breakpoints,
        include_lowest=True
    ).value_counts(sort=False)

    actual_counts = pd.cut(
        actual,
        bins=breakpoints,
        include_lowest=True
    ).value_counts(sort=False)

    expected_pct = (
        expected_counts / len(expected)
    ).clip(lower=0.0001)

    actual_pct = (
        actual_counts / len(actual)
    ).clip(lower=0.0001)

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return float(psi)


# -----------------------------------------
# FEATURE DRIFT
# -----------------------------------------

numeric_columns = X_train.select_dtypes(
    include=np.number
).columns

drift_results = []

for column in numeric_columns:

    psi = calculate_psi(
        X_train[column],
        X_test[column]
    )

    drift_results.append({

        "feature": column,

        "psi": psi

    })


drift_df = pd.DataFrame(drift_results)

drift_df["status"] = np.where(

    drift_df["psi"] < 0.10,

    "Stable",

    np.where(
        drift_df["psi"] < 0.20,
        "Monitor",
        "Investigate"
    )

)

drift_df.to_csv(
    TABLE_DIR / "feature_drift_report.csv",
    index=False
)

print(drift_df.sort_values(
    "psi",
    ascending=False
))


# -----------------------------------------
# MISSING VALUE MONITORING
# -----------------------------------------

missing_results = []

for column in X_test.columns:

    missing_results.append({

        "feature": column,

        "missing_rate": X_test[column].isna().mean()

    })


missing_df = pd.DataFrame(missing_results)

missing_df.to_csv(
    TABLE_DIR / "missing_rate_monitoring.csv",
    index=False
)

print("\nMissing value monitoring completed!")
