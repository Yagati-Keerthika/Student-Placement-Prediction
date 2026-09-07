import pandas as pd


def load_eda_dataset(filepath):
    return pd.read_csv(filepath)


def get_dataset_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "preview": df.head(10),
        "tail": df.tail(10),
        "missing": df.isnull().sum(),
        "describe": df.describe(include="all"),
        "numerical_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude="number").columns.tolist()
    }


def print_dataset_summary(df):
    summary = get_dataset_summary(df)

    print("Number of Rows:", summary["rows"])
    print("Number of Columns:", summary["columns"])
    print("\nFirst 10 Rows:")
    print(summary["preview"])
    print("\nLast 10 Rows:")
    print(summary["tail"])
    print("\nMissing Values:")
    print(summary["missing"])
    print("\nData Information:")
    df.info()
    print("\nStatistical Information:")
    print(summary["describe"])
    print("\nNumerical Columns:")
    print(summary["numerical_columns"])
    print("\nCategorical Columns:")
    print(summary["categorical_columns"])


if __name__ == "__main__":
    data_path = "uploads/student_placement_prediction_dataset_2026.csv"
    print_dataset_summary(load_eda_dataset(data_path))
