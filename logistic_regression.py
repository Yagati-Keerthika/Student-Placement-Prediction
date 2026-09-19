import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def build_logistic_regression_model(df, feature_columns, target_column):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = [
        column for column in feature_columns
        if column in numeric_columns and column != target_column
    ]

    if not selected_features:
        return None, {"error": "Please choose at least one valid numeric input feature."}, None

    if target_column not in df.columns:
        return None, {"error": "Selected target column does not exist."}, None

    model_df = df[selected_features + [target_column]].dropna().reset_index(drop=True)

    if model_df.empty:
        return None, {"error": "No valid rows are available after removing missing values."}, None

    X = model_df[selected_features]
    y = model_df[target_column]

    if y.nunique() != 2:
        return None, {"error": "The target feature must contain exactly two classes."}, None

    if not pd.api.types.is_numeric_dtype(y):
        target_values = y.astype("category")
        target_labels = list(target_values.cat.categories)
        y = target_values.cat.codes
    else:
        target_labels = sorted(y.unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(target_labels))))

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "coef": [float(value) for value in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "selected_features": selected_features,
        "target_column": target_column,
        "target_labels": target_labels
    }

    return model, metrics, target_labels


def predict_logistic_regression(model, prediction_values, feature_columns, target_labels):
    prediction_row = pd.DataFrame([prediction_values], columns=feature_columns)
    prediction_code = int(model.predict(prediction_row)[0])
    prediction_result = target_labels[prediction_code]
    prediction_probability = float(model.predict_proba(prediction_row).max())
    return prediction_result, prediction_probability
