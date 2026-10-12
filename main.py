import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from analysis_utils import get_analysis_feature_columns


# =========================================================
# TRAIN MACHINE LEARNING MODEL
# =========================================================

def train_model(df, target_column):

    # Separate input features

    feature_columns = get_analysis_feature_columns(df, target_column)
    X = df[feature_columns]


    # Target column

    y = df[target_column]


    # =====================================================
    # CONVERT CATEGORICAL DATA
    # =====================================================

    X = pd.get_dummies(
        X,
        drop_first=True
    )


    # =====================================================
    # HANDLE MISSING VALUES
    # =====================================================

    X = X.fillna(
        X.mean(numeric_only=True)
    )


    # =====================================================
    # CONVERT TARGET TO NUMERIC
    # =====================================================

    if y.dtype == "object":

        y = y.astype(
            "category"
        ).cat.codes


    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42
    )


    # =====================================================
    # CREATE MODEL
    # =====================================================

    model = LogisticRegression(
        max_iter=1000
    )


    # =====================================================
    # TRAIN MODEL
    # =====================================================

    model.fit(
        X_train,
        y_train
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    y_pred = model.predict(
        X_test
    )


    # =====================================================
    # ACCURACY
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    return model, accuracy


def prepare_prediction_page(df, submitted=False):

    target_column = next(
        (
            column for column in [
                "placement",
                "placed",
                "status",
                "placement_status"
            ]
            if column in df.columns
        ),
        None
    )

    page_data = {
        "prediction_result": None,
        "accuracy": None,
        "target_column": target_column,
        "error": None
    }

    if target_column is None:
        page_data["error"] = "Placement target column not found."
        return page_data

    if submitted:
        _, accuracy = train_model(df, target_column)
        page_data["prediction_result"] = "Model trained successfully."
        page_data["accuracy"] = accuracy

    return page_data