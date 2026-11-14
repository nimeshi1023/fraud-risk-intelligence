
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "outputs/models"
TABLE_DIR = ROOT / "outputs/tables"

splits = joblib.load(
    MODEL_DIR / "data_splits.joblib"
)

model = joblib.load(
    MODEL_DIR / "04_calibrated_model.joblib"
)

thresholds = joblib.load(
    MODEL_DIR / "selected_thresholds.joblib"
)

X_test = splits["X_test"]
y_test = splits["y_test"]

probabilities = model.predict_proba(
    X_test
)[:, 1]

# -----------------------------------------
# DEFAULT 0.50 POLICY
# -----------------------------------------

default_predictions = (
    probabilities >= 0.50
).astype(int)

print("DEFAULT THRESHOLD RESULTS")

print(classification_report(
    y_test,
    default_predictions,
    zero_division=0
))

# -----------------------------------------
# OPTIMIZED THREE-LEVEL POLICY
# -----------------------------------------

review_t = thresholds["review_threshold"]

block_t = thresholds["block_threshold"]

decisions = np.where(

    probabilities < review_t,

    "Approve",

    np.where(
        probabilities < block_t,
        "Review",
        "Block"
    )

)

# -----------------------------------------
# OPERATIONAL METRICS
# -----------------------------------------

fraud_capture = np.sum(
    (y_test == 1) & (decisions != "Approve")
)

false_positive_burden = np.sum(
    (y_test == 0) & (decisions != "Approve")
)

review_count = np.sum(decisions == "Review")

block_count = np.sum(decisions == "Block")

approve_count = np.sum(decisions == "Approve")

print("\nOPERATIONAL RESULTS")

print("Total transactions:", len(y_test))

print("Approve:", approve_count)

print("Review:", review_count)

print("Block:", block_count)

print("Fraud captured:", fraud_capture)

print("Fraud capture rate:", (
    fraud_capture / max(1, np.sum(y_test == 1))
))

print("False-positive burden:", false_positive_burden)

# -----------------------------------------
# FINAL BUSINESS COST
# -----------------------------------------

def business_cost(y, decisions):

    total = 0

    for actual, decision in zip(y, decisions):

        if decision == "Approve":

            if actual == 1:
                total += 5000

        elif decision == "Review":

            if actual == 1:
                total += 1100
            else:
                total += 100

        elif decision == "Block":

            if actual == 1:
                total += 20
            else:
                total += 500

    return total


optimized_cost = business_cost(
    y_test.to_numpy(),
    decisions
)

default_decisions = np.where(
    probabilities >= 0.50,
    "Block",
    "Approve"
)

default_cost = business_cost(
    y_test.to_numpy(),
    default_decisions
)

# -----------------------------------------
# METRIC REPORT
# -----------------------------------------

report = {

    "roc_auc": roc_auc_score(
        y_test, probabilities
    ),

    "pr_auc": average_precision_score(
        y_test, probabilities
    ),

    "brier_score": brier_score_loss(
        y_test, probabilities
    ),

    "precision_at_050": precision_score(
        y_test,
        default_predictions,
        zero_division=0
    ),

    "recall_at_050": recall_score(
        y_test,
        default_predictions,
        zero_division=0
    ),

    "f1_at_050": f1_score(
        y_test,
        default_predictions,
        zero_division=0
    ),

    "default_policy_cost": default_cost,

    "optimized_policy_cost": optimized_cost,

    "fraud_capture_rate": fraud_capture / max(
        1, np.sum(y_test == 1)
    ),

    "review_count": review_count,

    "block_count": block_count,

    "approve_count": approve_count

}

report_df = pd.DataFrame(
    [report]
)

report_df.to_csv(
    TABLE_DIR / "final_test_evaluation.csv",
    index=False
)

# -----------------------------------------
# CONFUSION MATRIX
# -----------------------------------------

cm = confusion_matrix(
    y_test,
    default_predictions
)

print("\nCONFUSION MATRIX")
print(cm)

print("\nFINAL TEST REPORT")
print(report_df.T)

# -----------------------------------------
# SAVE PREDICTIONS
# -----------------------------------------

prediction_df = pd.DataFrame({

    "actual_fraud": y_test.to_numpy(),

    "fraud_probability": probabilities,

    "decision": decisions

})

prediction_df.to_csv(
    TABLE_DIR / "final_test_predictions.csv",
    index=False
)

print("\nFINAL TEST EVALUATION COMPLETED!")
