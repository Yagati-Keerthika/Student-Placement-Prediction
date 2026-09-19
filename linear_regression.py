import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def build_linear_regression_model(df, feature_columns, target_column):
    df = df.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = [
        column for column in feature_columns
        if column in numeric_columns and column != target_column
    ]

    if not selected_features:
        return None, {"error": "Please choose at least one valid input feature."}

    if target_column not in df.columns:
        return None, {"error": "Selected target column does not exist."}

    model_df = df[selected_features + [target_column]].dropna().reset_index(drop=True)

    if model_df.empty:
        return None, {"error": "No valid rows are available after removing missing values."}

    X = model_df[selected_features]
    y = model_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    metrics = {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "coef": [float(value) for value in model.coef_],
        "intercept": float(model.intercept_),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "selected_features": selected_features,
        "target_column": target_column
    }

    return model, metrics


def predict_linear_regression(model, prediction_values, feature_columns):
    prediction_row = pd.DataFrame([prediction_values], columns=feature_columns)
    return float(model.predict(prediction_row)[0])
