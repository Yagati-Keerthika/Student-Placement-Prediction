import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


# =========================================================
# TRAIN MACHINE LEARNING MODEL
# =========================================================

def train_model(df, target_column):

    # Separate input features

    X = df.drop(
        columns=[target_column]
    )


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