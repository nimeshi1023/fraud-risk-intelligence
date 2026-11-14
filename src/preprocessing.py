
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


TARGET = "fraud"

DROP_COLUMNS = [
    "fraud",
    "transaction_id"
]


def prepare_features(df):

    X = df.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),

        (
            "scaler",
            StandardScaler()
        )

    ])

    categorical_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ])

    preprocessor = ColumnTransformer([

        (
            "numerical",
            numerical_pipeline,
            numeric_columns
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )

    ])

    return X, preprocessor


if __name__ == "__main__":

    print("Preprocessing module loaded successfully!")
