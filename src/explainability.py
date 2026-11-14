
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "outputs/models"
FIGURE_DIR = ROOT / "outputs/figures"
TABLE_DIR = ROOT / "outputs/tables"

base_model = joblib.load(
    MODEL_DIR / "03_fraud_model_pipeline.joblib"
)

splits = joblib.load(
    MODEL_DIR / "data_splits.joblib"
)

X_train = splits["X_train"]
X_val = splits["X_val"]

# -----------------------------------------
# PREPROCESS DATA
# -----------------------------------------

preprocessor = base_model.named_steps["preprocessor"]

classifier = base_model.named_steps["model"]

X_train_transformed = preprocessor.transform(X_train)

X_val_transformed = preprocessor.transform(X_val)

feature_names = preprocessor.get_feature_names_out()

# Convert sparse data to dense for SHAP.
# The dataset is small enough for this demonstration.

if hasattr(X_train_transformed, "toarray"):

    X_train_transformed = X_train_transformed.toarray()

if hasattr(X_val_transformed, "toarray"):

    X_val_transformed = X_val_transformed.toarray()


# -----------------------------------------
# BACKGROUND DATA
# -----------------------------------------

rng = np.random.default_rng(42)

background_indices = rng.choice(
    len(X_train_transformed),
    size=min(100, len(X_train_transformed)),
    replace=False
)

background = X_train_transformed[
    background_indices
]


# -----------------------------------------
# SHAP EXPLAINER
# -----------------------------------------

explainer = shap.Explainer(
    classifier,
    background,
    feature_names=feature_names
)

sample = X_val_transformed[
    :min(100, len(X_val_transformed))
]

shap_values = explainer(sample)


# -----------------------------------------
# GLOBAL FEATURE IMPORTANCE
# -----------------------------------------

values = np.asarray(shap_values.values)

if values.ndim == 3:

    values = values[:, :, 1]

importance = np.abs(values).mean(axis=0)

importance_df = pd.DataFrame({

    "feature": feature_names,

    "mean_absolute_shap": importance

}).sort_values(
    "mean_absolute_shap",
    ascending=False
)

importance_df.to_csv(
    TABLE_DIR / "shap_global_importance.csv",
    index=False
)

print(importance_df.head(20))


# -----------------------------------------
# GLOBAL SUMMARY PLOT
# -----------------------------------------

shap.summary_plot(
    shap_values,
    sample,
    feature_names=feature_names,
    show=False
)

plt.savefig(
    FIGURE_DIR / "14_shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------------------
# FEATURE IMPORTANCE BAR PLOT
# -----------------------------------------

plt.figure(figsize=(10, 8))

top_features = importance_df.head(15)

plt.barh(
    top_features["feature"],
    top_features["mean_absolute_shap"]
)

plt.gca().invert_yaxis()

plt.title("Top SHAP Features")

plt.xlabel("Mean Absolute SHAP Value")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "15_shap_importance.png",
    dpi=300
)

plt.close()


# -----------------------------------------
# FIVE LOCAL EXPLANATIONS
# -----------------------------------------

local_records = []

for i in range(min(5, len(X_val))):

    local_values = values[i]

    top_indices = np.argsort(
        np.abs(local_values)
    )[::-1][:5]

    explanation = []

    for index in top_indices:

        explanation.append({

            "feature": str(feature_names[index]),

            "shap_value": float(
                local_values[index]
            )

        })

    local_records.append({

        "transaction_number": i,

        "top_explanations": str(explanation)

    })


pd.DataFrame(local_records).to_csv(
    TABLE_DIR / "shap_local_explanations.csv",
    index=False
)

print("SHAP explanations completed!")
