import pandas as pd


def calculate_prediction_errors(df, feature_column="projects_count", target_column="salary_package_lpa", slope=2, intercept=3):
    errors = []

    for feature_value, target_value in zip(df[feature_column], df[target_column]):
        prediction = slope * feature_value + intercept
        errors.append({
            "feature": feature_value,
            "actual": target_value,
            "predicted": prediction,
            "error": target_value - prediction
        })

    return pd.DataFrame(errors)


def load_error_prediction_data(filepath):
    if filepath.lower().endswith(".xlsx"):
        return pd.read_excel(filepath)

    return pd.read_csv(filepath)


if __name__ == "__main__":
    data_path = "uploads/student_placement_prediction_dataset_2026.csv"
    results = calculate_prediction_errors(load_error_prediction_data(data_path))
    print(results.to_string(index=False))