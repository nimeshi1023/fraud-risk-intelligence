
import pandas as pd
import numpy as np


def create_features(df):

    data = df.copy()

    # -----------------------------------------
    # 1. Transaction velocity ratio
    # -----------------------------------------

    data["velocity_ratio"] = (
        data["transactions_last_1h"] /
        (data["transactions_last_24h"] + 1)
    )

    # -----------------------------------------
    # 2. Amount per recent transaction
    # -----------------------------------------

    data["amount_per_recent_transaction"] = (
        data["transaction_amount"] /
        (data["transactions_last_24h"] + 1)
    )

    # -----------------------------------------
    # 3. Transaction amount relative to account age
    # -----------------------------------------

    data["amount_account_age_ratio"] = (
        data["transaction_amount"] /
        (data["account_age_days"] + 1)
    )

    # -----------------------------------------
    # 4. Combined risk score
    # -----------------------------------------

    data["combined_risk_score"] = (
        data["merchant_risk_score"] +
        data["ip_risk_score"]
    ) / 2

    # -----------------------------------------
    # 5. Maximum risk score
    # -----------------------------------------

    data["maximum_risk_score"] = data[
        ["merchant_risk_score", "ip_risk_score"]
    ].max(axis=1)

    # -----------------------------------------
    # 6. Amount and merchant risk interaction
    # -----------------------------------------

    data["amount_merchant_interaction"] = (
        data["transaction_amount"] *
        data["merchant_risk_score"]
    )

    # -----------------------------------------
    # 7. New device and distance interaction
    # -----------------------------------------

    data["new_device_distance"] = (
        data["new_device"] *
        data["distance_from_usual_location_km"]
    )

    # -----------------------------------------
    # 8. Failed login intensity
    # -----------------------------------------

    data["failed_login_intensity"] = (
        data["failed_logins_24h"] /
        (data["transactions_last_24h"] + 1)
    )

    # -----------------------------------------
    # 9. Account and device age difference
    # -----------------------------------------

    data["account_device_age_gap"] = (
        data["account_age_days"] -
        data["device_age_days"]
    )

    # -----------------------------------------
    # 10. Night transaction indicator
    # -----------------------------------------

    data["night_transaction"] = (
        (data["hour"] >= 0) &
        (data["hour"] <= 5)
    ).astype(int)

    # -----------------------------------------
    # 11. Evening transaction indicator
    # -----------------------------------------

    data["evening_transaction"] = (
        (data["hour"] >= 18) &
        (data["hour"] <= 23)
    ).astype(int)

    # -----------------------------------------
    # 12. Weekend indicator
    # -----------------------------------------

    data["weekend_transaction"] = (
        data["day_of_week"].isin([5, 6])
    ).astype(int)

    # -----------------------------------------
    # 13. Rapid transaction indicator
    # -----------------------------------------

    data["high_velocity"] = (
        data["transactions_last_1h"] >= 5
    ).astype(int)

    # -----------------------------------------
    # 14. Rapid login to transaction
    # -----------------------------------------

    data["rapid_transaction_after_login"] = (
        data["login_to_transaction_minutes"] <= 5
    ).fillna(False).astype(int)

    # -----------------------------------------
    # 15. Authentication and channel interaction
    # -----------------------------------------

    data["authentication_channel"] = (
        data["authentication_level"]
        .fillna("Unknown").astype(str)
        + "_"
        + data["channel"].astype(str)
    )

    # -----------------------------------------
    # 16. Country and channel interaction
    # -----------------------------------------

    data["country_channel"] = (
        data["country"].astype(str)
        + "_"
        + data["channel"].astype(str)
    )

    # -----------------------------------------
    # 17. Amount log transformation
    # -----------------------------------------

    data["log_transaction_amount"] = np.log1p(
        data["transaction_amount"].clip(lower=0)
    )

    return data


if __name__ == "__main__":

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    input_path = (
        root / "data" / "raw" /
        "transaction_fraud_risk_advanced_dataset.csv"
    )

    output_path = (
        root / "data" / "processed" /
        "02_processed_dataset.csv"
    )

    df = pd.read_csv(input_path)

    processed = create_features(df)

    processed.to_csv(output_path, index=False)

    print("Feature engineering completed!")
    print("Original shape:", df.shape)
    print("New shape:", processed.shape)
