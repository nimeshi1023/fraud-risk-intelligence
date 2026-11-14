
import pandas as pd
import numpy as np
import os

# ------------------------------------------------
# 1. PATH CONFIGURATION
# ------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "raw",
    "transaction_fraud_risk_advanced_dataset.csv"
)

OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "tables")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------

def load_data():

    df = pd.read_csv(DATA_PATH)

    print("\n========== DATASET OVERVIEW ==========")

    print("Dataset shape:", df.shape)

    print("\nFirst five rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    return df


# ------------------------------------------------
# 3. DATA TYPE AUDIT
# ------------------------------------------------

def data_type_audit(df):

    report = pd.DataFrame({
        "column": df.columns,
        "data_type": df.dtypes.astype(str).values,
        "missing_count": df.isnull().sum().values,
        "missing_percentage": (
            df.isnull().mean().values * 100
        ),
        "unique_values": df.nunique().values
    })

    report.to_csv(
        os.path.join(OUTPUT_DIR, "data_type_audit.csv"),
        index=False
    )

    print("\n========== DATA TYPES ==========")
    print(report.to_string(index=False))

    return report


# ------------------------------------------------
# 4. DUPLICATE AUDIT
# ------------------------------------------------

def duplicate_audit(df):

    total_duplicates = df.duplicated().sum()

    duplicate_ids = df["transaction_id"].duplicated().sum()

    print("\n========== DUPLICATE AUDIT ==========")

    print("Duplicate rows:", total_duplicates)

    print("Duplicate transaction IDs:", duplicate_ids)

    return total_duplicates, duplicate_ids


# ------------------------------------------------
# 5. MISSING VALUE AUDIT
# ------------------------------------------------

def missing_value_audit(df):

    missing_report = pd.DataFrame({

        "missing_count": df.isnull().sum(),

        "missing_percentage": (
            df.isnull().mean() * 100
        )

    })

    missing_report = missing_report.sort_values(
        by="missing_percentage",
        ascending=False
    )

    missing_report.to_csv(
        os.path.join(OUTPUT_DIR, "missing_value_report.csv")
    )

    print("\n========== MISSING VALUES ==========")

    print(missing_report)

    return missing_report


# ------------------------------------------------
# 6. NUMERICAL DISTRIBUTION AUDIT
# ------------------------------------------------

def numerical_audit(df):

    numerical_df = df.select_dtypes(
        include=np.number
    )

    numerical_report = numerical_df.describe().T

    numerical_report["skewness"] = numerical_df.skew()

    numerical_report["missing_count"] = (
        numerical_df.isnull().sum()
    )

    numerical_report.to_csv(
        os.path.join(OUTPUT_DIR, "numerical_audit.csv")
    )

    print("\n========== NUMERICAL AUDIT ==========")

    print(numerical_report)

    return numerical_report


# ------------------------------------------------
# 7. CATEGORICAL CARDINALITY AUDIT
# ------------------------------------------------

def categorical_audit(df):

    categorical_df = df.select_dtypes(
        include=["object", "category"]
    )

    records = []

    for column in categorical_df.columns:

        records.append({

            "column": column,

            "unique_count": df[column].nunique(),

            "most_frequent": (
                df[column].mode().iloc[0]
                if not df[column].mode().empty
                else None
            )

        })

    report = pd.DataFrame(records)

    report.to_csv(
        os.path.join(OUTPUT_DIR, "categorical_audit.csv"),
        index=False
    )

    print("\n========== CATEGORICAL AUDIT ==========")

    print(report)

    return report


# ------------------------------------------------
# 8. IMPOSSIBLE VALUE CHECKS
# ------------------------------------------------

def impossible_value_audit(df):

    checks = {

        "transaction_amount_negative":
            df["transaction_amount"] < 0,

        "distance_negative":
            df["distance_from_usual_location_km"] < 0,

        "transactions_last_1h_negative":
            df["transactions_last_1h"] < 0,

        "transactions_last_24h_negative":
            df["transactions_last_24h"] < 0,

        "account_age_negative":
            df["account_age_days"] < 0,

        "device_age_negative":
            df["device_age_days"] < 0,

        "email_age_negative":
            df["email_age_days"] < 0,

        "hour_invalid":
            ~df["hour"].between(0, 23),

        "day_of_week_invalid":
            ~df["day_of_week"].between(0, 6),

        "new_device_invalid":
            ~df["new_device"].isin([0, 1]),

        "billing_mismatch_invalid":
            ~df["billing_address_mismatch"].isin([0, 1]),

        "fraud_invalid":
            ~df["fraud"].isin([0, 1])

    }

    results = []

    for name, condition in checks.items():

        count = condition.fillna(False).sum()

        results.append({

            "check": name,

            "invalid_count": count

        })

    report = pd.DataFrame(results)

    report.to_csv(
        os.path.join(OUTPUT_DIR, "impossible_value_report.csv"),
        index=False
    )

    print("\n========== IMPOSSIBLE VALUE CHECK ==========")

    print(report)

    return report


# ------------------------------------------------
# 9. MISSINGNESS VS FRAUD
# ------------------------------------------------

def missingness_fraud_analysis(df):

    results = []

    for column in df.columns:

        if df[column].isnull().sum() > 0:

            missing_indicator = df[column].isnull()

            fraud_rate_missing = df.loc[
                missing_indicator, "fraud"
            ].mean()

            fraud_rate_present = df.loc[
                ~missing_indicator, "fraud"
            ].mean()

            results.append({

                "feature": column,

                "missing_count": missing_indicator.sum(),

                "fraud_rate_when_missing":
                    fraud_rate_missing,

                "fraud_rate_when_present":
                    fraud_rate_present

            })

    report = pd.DataFrame(results)

    report.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "missingness_fraud_analysis.csv"
        ),
        index=False
    )

    print("\n========== MISSINGNESS VS FRAUD ==========")

    print(report)

    return report


# ------------------------------------------------
# 10. COMPLETE AUDIT
# ------------------------------------------------

def main():

    df = load_data()

    data_type_audit(df)

    duplicate_audit(df)

    missing_value_audit(df)

    numerical_audit(df)

    categorical_audit(df)

    impossible_value_audit(df)

    missingness_fraud_analysis(df)

    print("\nDATA AUDIT COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":

    main()
