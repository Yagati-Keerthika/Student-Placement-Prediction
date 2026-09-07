import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def build_linear_regression_model(df, feature_columns, target_column):

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    selected_features = [
        column for column in feature_columns
        if column in numeric_columns and column != target_column
    ]

    if not selected_features:
        return None, {
            "error": "Please choose at least one valid input feature."
        }, None, None, None, None

    if target_column not in df.columns:
        return None, {
            "error": "Selected target column does not exist."
        }, None, None, None, None

    model_df = df[selected_features + [target_column]].dropna().reset_index(drop=True)

    if model_df.empty:
        return None, {
            "error": "No valid rows are available after removing missing values."
        }, None, None, None, None

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

    return model, metrics, X_train, X_test, y_train, y_test


# =========================================================
# LOAD PREPROCESSED DATA
# =========================================================

def load_preprocessed_data(filepath):

    df = pd.read_csv(filepath)

    return df


# =========================================================
# TRAIN LINEAR REGRESSION MODEL
# =========================================================

def train_linear_regression_model(df):

    df = df.copy()

    # Use only students who got placed because salary is 0 for not placed
    df = df[df["salary_package_lpa"] > 0].reset_index(drop=True)

    if df.empty:
        raise ValueError("No placed students available for salary prediction.")

    X = df[["cgpa"]]
    y = df["salary_package_lpa"]

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
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("========================================================")
    print("LINEAR REGRESSION MODEL TRAINING")
    print("========================================================")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print()

    coef = model.coef_[0]
    intercept = model.intercept_

    print("Coefficient (slope):", coef)
    print("Intercept:", intercept)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-Square (R2):", r2)
    print("========================================================")

    return model, X_train, y_train, X_test, y_test, mse, rmse, mae, r2, coef, intercept


# =========================================================
# PREDICT EXPECTED SALARY
# =========================================================

def predict_salary(model, cgpa):

    cgpa = np.array([[cgpa]])
    predicted_salary = model.predict(cgpa)

    return predicted_salary[0]


def predict_linear_regression(model, prediction_values, feature_columns):

    prediction_row = pd.DataFrame(
        [prediction_values],
        columns=feature_columns
    )

    return float(model.predict(prediction_row)[0])


def prepare_linear_regression_page(df, form_data=None, submitted=False):

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    default_target = (
        "salary_package_lpa"
        if "salary_package_lpa" in numeric_columns
        else numeric_columns[0] if numeric_columns else None
    )
    default_features = ["cgpa"] if "cgpa" in numeric_columns else numeric_columns[:1]

    page_data = {
        "numeric_columns": numeric_columns,
        "selected_features": default_features,
        "target_column": default_target,
        "metrics": None,
        "prediction_result": None,
        "error": None
    }

    if not numeric_columns:
        page_data["error"] = "This dataset does not contain numeric columns for regression."
        return page_data

    if not submitted:
        return page_data

    form_data = form_data or {}
    selected_features = form_data.getlist("feature")
    target_column = form_data.get("target_feature", default_target)
    page_data.update({
        "selected_features": selected_features,
        "target_column": target_column
    })

    if not selected_features:
        page_data["error"] = "Please select at least one input feature."
        return page_data

    if target_column not in numeric_columns:
        page_data["error"] = "Please select a valid numeric target feature."
        return page_data

    model, metrics, _, _, _, _ = build_linear_regression_model(
        df,
        selected_features,
        target_column
    )

    if model is None:
        page_data["error"] = metrics.get("error")
        return page_data

    page_data["metrics"] = metrics
    prediction_values = {}

    for feature in selected_features:
        value = form_data.get(feature, "")
        if value == "":
            page_data["error"] = f"Please enter a value for {feature}."
            return page_data
        try:
            prediction_values[feature] = float(value)
        except (TypeError, ValueError):
            page_data["error"] = f"{feature} must be a valid number."
            return page_data

    page_data["prediction_result"] = predict_linear_regression(
        model,
        prediction_values,
        selected_features
    )
    return page_data


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    data_path = "processed/encoded_student_placement_prediction_dataset_2026.csv"

    df = load_preprocessed_data(data_path)

    model, X_train, y_train, X_test, y_test, mse, rmse, mae, r2, coef, intercept = train_linear_regression_model(df)

    student_cgpa = float(input("Enter student CGPA: "))
    predicted_salary = predict_salary(model, student_cgpa)

    print()
    print("Coefficient (model.coef[0]):", coef)
    print("Intercept:", intercept)
    print("Predicted Salary for CGPA", student_cgpa, "is:", round(predicted_salary, 2), "LPA")
