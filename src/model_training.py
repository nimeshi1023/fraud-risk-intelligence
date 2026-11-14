
import os
import sys
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)

from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

sys.path.append(str(Path(__file__).resolve().parent))

from feature_engineering import create_features
from preprocessing import prepare_features


# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/raw/transaction_fraud_risk_advanced_dataset.csv"

MODEL_DIR = ROOT / "outputs/models"
TABLE_DIR = ROOT / "outputs/tables"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

df = pd.read_csv(DATA_PATH)

df = create_features(df)

X, preprocessor = prepare_features(df)

y = df["fraud"]


# ------------------------------------------------
# TRAIN / VALIDATION / TEST SPLIT
# ------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ------------------------------------------------
# SAVE SPLITS FOR REPRODUCIBILITY
# ------------------------------------------------

joblib.dump(
    {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    },
    MODEL_DIR / "data_splits.joblib"
)


# ------------------------------------------------
# CROSS VALIDATION
# ------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ------------------------------------------------
# DEFINE MODEL PIPELINES
# ------------------------------------------------

models = {

    "Logistic Regression": {

        "pipeline": Pipeline([

            ("preprocessor", preprocessor),

            ("sampler", "passthrough"),

            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))

        ]),

        "params": {

            "model__C": [0.01, 0.1, 1, 10],

            "model__solver": ["lbfgs"]

        }

    },


    "Random Forest": {

        "pipeline": Pipeline([

            ("preprocessor", preprocessor),

            ("sampler", "passthrough"),

            ("model", RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))

        ]),

        "params": {

            "model__n_estimators": [100, 200, 300],

            "model__max_depth": [5, 10, None],

            "model__min_samples_leaf": [1, 3, 5],

            "model__max_features": ["sqrt", 0.7]

        }

    },


    "Random Forest ROS": {

        "pipeline": Pipeline([

            ("preprocessor", preprocessor),

            ("sampler", RandomOverSampler(
                random_state=RANDOM_STATE
            )),

            ("model", RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))

        ]),

        "params": {

            "model__n_estimators": [100, 200],

            "model__max_depth": [5, 10, None],

            "model__min_samples_leaf": [1, 3, 5]

        }

    },


    "XGBoost": {

        "pipeline": Pipeline([

            ("preprocessor", preprocessor),

            ("sampler", "passthrough"),

            ("model", XGBClassifier(

                n_estimators=200,

                max_depth=4,

                learning_rate=0.05,

                subsample=0.8,

                colsample_bytree=0.8,

                scale_pos_weight=(
                    (y_train == 0).sum() /
                    (y_train == 1).sum()
                ),

                eval_metric="logloss",

                random_state=RANDOM_STATE,

                n_jobs=-1

            ))

        ]),

        "params": {

            "model__n_estimators": [100, 200, 300],

            "model__max_depth": [3, 4, 6],

            "model__learning_rate": [0.01, 0.05, 0.1],

            "model__subsample": [0.7, 0.8, 1.0]

        }

    }

}


# ------------------------------------------------
# MODEL TRAINING AND OPTIMIZATION
# ------------------------------------------------

results = []

best_models = {}

for model_name, config in models.items():

    print("\nTraining:", model_name)

    search = RandomizedSearchCV(

        estimator=config["pipeline"],

        param_distributions=config["params"],

        n_iter=8,

        scoring="average_precision",

        cv=cv,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        refit=True,

        verbose=1

    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    best_models[model_name] = best_model

    val_probabilities = best_model.predict_proba(X_val)[:, 1]

    val_predictions = (
        val_probabilities >= 0.50
    ).astype(int)

    results.append({

        "model": model_name,

        "cv_pr_auc": search.best_score_,

        "validation_roc_auc": roc_auc_score(
            y_val, val_probabilities
        ),

        "validation_pr_auc": average_precision_score(
            y_val, val_probabilities
        ),

        "precision": precision_score(
            y_val, val_predictions, zero_division=0
        ),

        "recall": recall_score(
            y_val, val_predictions, zero_division=0
        ),

        "f1": f1_score(
            y_val, val_predictions, zero_division=0
        ),

        "best_parameters": str(search.best_params_)

    })

    joblib.dump(

        best_model,

        MODEL_DIR / (
            model_name.lower().replace(" ", "_") + ".joblib"
        )

    )


# ------------------------------------------------
# COMPARE MODELS
# ------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    TABLE_DIR / "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")

print(
    results_df.sort_values(
        "validation_pr_auc",
        ascending=False
    )
)


# ------------------------------------------------
# SELECT BEST MODEL USING VALIDATION PR-AUC
# ------------------------------------------------

best_name = results_df.sort_values(
    "validation_pr_auc",
    ascending=False
).iloc[0]["model"]

best_model = best_models[best_name]

joblib.dump(
    best_model,
    MODEL_DIR / "03_fraud_model_pipeline.joblib"
)

joblib.dump(
    best_name,
    MODEL_DIR / "selected_model_name.joblib"
)

print("\nSelected model:", best_name)

print("\nMODEL TRAINING COMPLETED!")

