
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from feature_engineering import create_features


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "outputs/models"
OUTPUT_DIR = ROOT / "outputs/tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# LOAD PRODUCTION CANDIDATES
# -----------------------------------------

model = joblib.load(
    MODEL_DIR / "04_calibrated_model.joblib"
)

thresholds = joblib.load(
    MODEL_DIR / "selected_thresholds.joblib"
)

review_threshold = thresholds["review_threshold"]

block_threshold = thresholds["block_threshold"]


# -----------------------------------------
# DECISION FUNCTION
# -----------------------------------------

def fraud_decision(probability):

    if probability < review_threshold:

        return "Approve"

    elif probability < block_threshold:

        return "Review"

    else:

        return "Block"


# -----------------------------------------
# EXPLANATION FUNCTION
# -----------------------------------------

def generate_reason(row):

    reasons = []

    if row.get("new_device", 0) == 1:
        reasons.append("Transaction from a new device")

    distance = row.get(
        "distance_from_usual_location_km"
    )

    if pd.notna(distance) and distance > 50:
        reasons.append("Unusual transaction distance")

    velocity = row.get("transactions_last_1h", 0)

    if velocity >= 5:
        reasons.append("High transaction velocity")

    failed = row.get("failed_logins_24h", 0)

    if failed >= 3:
        reasons.append("Multiple failed login attempts")

    merchant_risk = row.get("merchant_risk_score")

    if pd.notna(merchant_risk) and merchant_risk >= 0.7:
        reasons.append("High merchant risk score")

    ip_risk = row.get("ip_risk_score")

    if pd.notna(ip_risk) and ip_risk >= 0.7:
        reasons.append("High IP risk score")

    if row.get("billing_address_mismatch", 0) == 1:
        reasons.append("Billing address mismatch")

    if not reasons:
        reasons.append(
            "No predefined high-risk rule was triggered"
        )

    return "; ".join(reasons)


# -----------------------------------------
# SINGLE TRANSACTION PREDICTION
# -----------------------------------------

def predict_transaction(transaction):

    raw_df = pd.DataFrame([transaction])

    engineered = create_features(raw_df)

    probability = model.predict_proba(
        engineered.drop(
            columns=["fraud", "transaction_id"],
            errors="ignore"
        )
    )[:, 1][0]

    decision = fraud_decision(probability)

    explanation = generate_reason(
        transaction
    )

    return {

        "fraud_probability": round(
            float(probability), 4
        ),

        "decision": decision,

        "explanation": explanation

    }


# -----------------------------------------
# VALIDATION BATCH SIMULATION
# -----------------------------------------

splits = joblib.load(
    MODEL_DIR / "data_splits.joblib"
)

X_val = splits["X_val"]

y_val = splits["y_val"]

probabilities = model.predict_proba(
    X_val
)[:, 1]

decisions = [
    fraud_decision(p)
    for p in probabilities
]

results = X_val.copy()

results["actual_fraud"] = y_val.values

results["fraud_probability"] = probabilities

results["decision"] = decisions

results["explanation"] = [
    generate_reason(row)
    for row in results.to_dict("records")
]

results.to_csv(
    OUTPUT_DIR / "fraud_decision_results.csv",
    index=False
)

print(results[
    [
        "fraud_probability",
        "decision",
        "explanation"
    ]
].head(10))

print("\nDECISION DISTRIBUTION")

print(results["decision"].value_counts())

print("\nDecision engine completed!")
