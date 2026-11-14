
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve
)

from sklearn.metrics import (
    brier_score_loss,
    log_loss
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "outputs/models"
TABLE_DIR = ROOT / "outputs/tables"
FIGURE_DIR = ROOT / "outputs/figures"

splits = joblib.load(
    MODEL_DIR / "data_splits.joblib"
)

X_train = splits["X_train"]
X_val = splits["X_val"]

y_train = splits["y_train"]
y_val = splits["y_val"]

base_model = joblib.load(
    MODEL_DIR / "03_fraud_model_pipeline.joblib"
)


# -----------------------------------------
# FIT CALIBRATION MODELS
# -----------------------------------------

calibrators = {}

methods = ["sigmoid", "isotonic"]

results = []

for method in methods:

    print("Fitting:", method)

    calibrated = CalibratedClassifierCV(

        estimator=base_model,

        method=method,

        cv=5

    )

    calibrated.fit(X_train, y_train)

    probabilities = calibrated.predict_proba(X_val)[:, 1]

    brier = brier_score_loss(
        y_val,
        probabilities
    )

    loss = log_loss(
        y_val,
        probabilities
    )

    results.append({

        "method": method,

        "brier_score": brier,

        "log_loss": loss

    })

    calibrators[method] = calibrated

    joblib.dump(
        calibrated,
        MODEL_DIR / f"calibrated_{method}.joblib"
    )


# -----------------------------------------
# COMPARE CALIBRATION
# -----------------------------------------

comparison = pd.DataFrame(results)

comparison.to_csv(
    TABLE_DIR / "calibration_comparison.csv",
    index=False
)

print(comparison)


# -----------------------------------------
# SELECT BEST CALIBRATOR
# -----------------------------------------

best_method = comparison.sort_values(
    "brier_score"
).iloc[0]["method"]

best_calibrated = calibrators[best_method]

joblib.dump(
    best_calibrated,
    MODEL_DIR / "04_calibrated_model.joblib"
)

print("Selected calibration:", best_method)


# -----------------------------------------
# CALIBRATION CURVE
# -----------------------------------------

plt.figure(figsize=(9, 7))

for method in methods:

    model = calibrators[method]

    probabilities = model.predict_proba(X_val)[:, 1]

    fraction_positive, mean_predicted = calibration_curve(

        y_val,

        probabilities,

        n_bins=10,

        strategy="quantile"

    )

    plt.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label=method
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel("Mean Predicted Probability")

plt.ylabel("Actual Fraud Rate")

plt.title("Probability Calibration Comparison")

plt.legend()

plt.savefig(
    FIGURE_DIR / "13_calibration_curve.png",
    dpi=300
)

plt.show()
