
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "outputs/models"
TABLE_DIR = ROOT / "outputs/tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# LOAD VALIDATION DATA
# -----------------------------------------

splits = joblib.load(
    MODEL_DIR / "data_splits.joblib"
)

X_val = splits["X_val"]
y_val = splits["y_val"].to_numpy()

calibrated_model = joblib.load(
    MODEL_DIR / "04_calibrated_model.joblib"
)

probabilities = calibrated_model.predict_proba(
    X_val
)[:, 1]


# -----------------------------------------
# COST ASSUMPTIONS
# -----------------------------------------

def calculate_cost(
    y_true,
    decisions,
    fraud_loss=5000,
    review_fee=100,
    false_block_cost=500,
    fraud_block_cost=20,
    review_detection=0.80
):

    y_true = np.asarray(y_true)
    decisions = np.asarray(decisions)

    total_cost = 0

    for actual, decision in zip(y_true, decisions):

        if decision == "Approve":

            if actual == 1:
                total_cost += fraud_loss

        elif decision == "Review":

            if actual == 1:

                total_cost += (
                    review_fee
                    + (1 - review_detection) * fraud_loss
                )

            else:

                total_cost += review_fee

        elif decision == "Block":

            if actual == 1:

                total_cost += fraud_block_cost

            else:

                total_cost += false_block_cost

    return total_cost


# -----------------------------------------
# THREE-LEVEL DECISION
# -----------------------------------------

def make_decisions(
    probabilities,
    review_threshold,
    block_threshold
):

    decisions = []

    for probability in probabilities:

        if probability < review_threshold:

            decisions.append("Approve")

        elif probability < block_threshold:

            decisions.append("Review")

        else:

            decisions.append("Block")

    return np.array(decisions)


# -----------------------------------------
# THRESHOLD OPTIMIZATION
# -----------------------------------------

thresholds = np.arange(
    0.05,
    0.96,
    0.01
)

best_result = None

all_results = []

for review_t in thresholds:

    for block_t in thresholds:

        if review_t >= block_t:
            continue

        decisions = make_decisions(
            probabilities,
            review_t,
            block_t
        )

        cost = calculate_cost(
            y_val,
            decisions
        )

        result = {

            "review_threshold": review_t,

            "block_threshold": block_t,

            "expected_cost": cost,

            "cost_per_transaction": cost / len(y_val),

            "review_count": np.sum(decisions == "Review"),

            "block_count": np.sum(decisions == "Block"),

            "approve_count": np.sum(decisions == "Approve")

        }

        all_results.append(result)

        if (
            best_result is None
            or cost < best_result["expected_cost"]
        ):

            best_result = result


# -----------------------------------------
# SAVE THRESHOLD RESULTS
# -----------------------------------------

threshold_df = pd.DataFrame(all_results)

threshold_df.to_csv(
    TABLE_DIR / "05_threshold_cost_analysis.csv",
    index=False
)

print("\nBEST THRESHOLDS")

print(best_result)


# -----------------------------------------
# SAVE SELECTED THRESHOLDS
# -----------------------------------------

joblib.dump(
    best_result,
    MODEL_DIR / "selected_thresholds.joblib"
)


# -----------------------------------------
# COMPARE DEFAULT 0.50
# -----------------------------------------

default_predictions = np.where(
    probabilities >= 0.50,
    "Block",
    "Approve"
)

default_cost = calculate_cost(
    y_val,
    default_predictions
)

print("\nDEFAULT 0.50 POLICY COST:", default_cost)

print(
    "OPTIMIZED POLICY COST:",
    best_result["expected_cost"]
)


# -----------------------------------------
# SENSITIVITY ANALYSIS
# -----------------------------------------

scenarios = [

    {
        "scenario": "Low fraud loss",
        "fraud_loss": 2000,
        "false_block_cost": 500,
        "review_fee": 100
    },

    {
        "scenario": "Base case",
        "fraud_loss": 5000,
        "false_block_cost": 500,
        "review_fee": 100
    },

    {
        "scenario": "High fraud loss",
        "fraud_loss": 10000,
        "false_block_cost": 500,
        "review_fee": 100
    },

    {
        "scenario": "High customer friction",
        "fraud_loss": 5000,
        "false_block_cost": 1500,
        "review_fee": 300
    }

]

sensitivity_results = []

for scenario in scenarios:

    best_scenario = None

    for review_t in thresholds:

        for block_t in thresholds:

            if review_t >= block_t:
                continue

            decisions = make_decisions(
                probabilities,
                review_t,
                block_t
            )

            cost = calculate_cost(

                y_val,

                decisions,

                fraud_loss=scenario["fraud_loss"],

                false_block_cost=scenario["false_block_cost"],

                review_fee=scenario["review_fee"]

            )

            if (
                best_scenario is None
                or cost < best_scenario["cost"]
            ):

                best_scenario = {

                    "review_threshold": review_t,

                    "block_threshold": block_t,

                    "cost": cost

                }

    sensitivity_results.append({

        "scenario": scenario["scenario"],

        **best_scenario

    })


sensitivity_df = pd.DataFrame(
    sensitivity_results
)

sensitivity_df.to_csv(
    TABLE_DIR / "cost_sensitivity.csv",
    index=False
)

print("\nCOST SENSITIVITY")

print(sensitivity_df)
