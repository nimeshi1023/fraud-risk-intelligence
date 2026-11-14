
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "outputs/models"

TABLE_DIR = ROOT / "outputs/tables"

model = joblib.load(
    MODEL_DIR / "04_calibrated_model.joblib"
)

splits = joblib.load(
    MODEL_DIR / "data_splits.joblib"
)

X_test = splits["X_test"].copy()

y_test = splits["y_test"]


# -----------------------------------------
# BASELINE
# -----------------------------------------

baseline_prob = model.predict_proba(
    X_test
)[:, 1]


def evaluate_scenario(name, data):

    probabilities = model.predict_proba(
        data
    )[:, 1]

    return {

        "scenario": name,

        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y_test,
            probabilities
        ),

        "mean_risk_probability": probabilities.mean()

    }


results = []

results.append(
    evaluate_scenario("Baseline", X_test)
)


# -----------------------------------------
# 1. INCREASE TRANSACTION AMOUNT
# -----------------------------------------

amount_shift = X_test.copy()

amount_shift["transaction_amount"] *= 1.25

results.append(
    evaluate_scenario(
        "Transaction amount +25%",
        amount_shift
    )
)


# -----------------------------------------
# 2. INCREASE TRANSACTION VELOCITY
# -----------------------------------------

velocity_shift = X_test.copy()

velocity_shift["transactions_last_1h"] = (
    velocity_shift["transactions_last_1h"] * 1.5
).round()

results.append(
    evaluate_scenario(
        "Transaction velocity +50%",
        velocity_shift
    )
)


# -----------------------------------------
# 3. INCREASE MISSING VALUES
# -----------------------------------------

missing_shift = X_test.copy()

rng = np.random.default_rng(42)

for column in [
    "transaction_amount",
    "distance_from_usual_location_km",
    "merchant_risk_score",
    "ip_risk_score"
]:

    mask = rng.random(len(missing_shift)) < 0.10

    missing_shift.loc[mask, column] = np.nan


results.append(
    evaluate_scenario(
        "10% additional missingness",
        missing_shift
    )
)


# -----------------------------------------
# 4. EXTREME OBSERVATIONS
# -----------------------------------------

extreme_shift = X_test.copy()

extreme_shift["transaction_amount"] *= 5

results.append(
    evaluate_scenario(
        "Extreme transaction amounts",
        extreme_shift
    )
)


# -----------------------------------------
# 5. CHANNEL MIX SHIFT
# -----------------------------------------

channel_shift = X_test.copy()

mask = rng.random(len(channel_shift)) < 0.25

channel_shift.loc[mask, "channel"] = "E-commerce"

results.append(
    evaluate_scenario(
        "Channel mix shift",
        channel_shift
    )
)


# -----------------------------------------
# SAVE RESULTS
# -----------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    TABLE_DIR / "robustness_results.csv",
    index=False
)

print(results_df)
