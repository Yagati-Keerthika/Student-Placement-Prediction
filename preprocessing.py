import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer
)

# =========================================================
# READ DATASET
# =========================================================

def load_dataset(filepath):

    df = pd.read_csv(filepath)

    return df


# =========================================================
# GET NUMERICAL COLUMNS
# =========================================================

def get_numerical_columns(df):

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    return numerical_columns


# =========================================================
# GET CATEGORICAL COLUMNS
# =========================================================

def get_categorical_columns(df):

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    return categorical_columns


# =========================================================
# MISSING VALUE INFORMATION
# =========================================================

def get_missing_values(df):

    missing = df.isnull().sum().to_frame(
        name="Missing Values"
    )

    missing.index.name = "Column"

    return missing


# =========================================================
# GET MISSING COLUMNS
# =========================================================

def get_missing_columns(df):

    return df.columns[
        df.isnull().any()
    ].tolist()


# =========================================================
# FILL MISSING VALUES BY STRATEGY
# =========================================================

def fill_missing_values(df, method, columns=None):

    if columns is None:

        columns = get_missing_columns(df)

    for column in columns:

        if column not in df.columns:

            continue

        if df[column].isnull().sum() == 0:

            continue

        if method == "mean":

            if pd.api.types.is_numeric_dtype(df[column]):

                df[column] = df[column].fillna(
                    df[column].mean()
                )

            else:

                df[column] = df[column].fillna(
                    df[column].mode().iloc[0]
                )

        elif method == "median":

            if pd.api.types.is_numeric_dtype(df[column]):

                df[column] = df[column].fillna(
                    df[column].median()
                )

            else:

                df[column] = df[column].fillna(
                    df[column].mode().iloc[0]
                )

        elif method == "mode":

            mode_value = df[column].mode(dropna=True)

            if not mode_value.empty:

                df[column] = df[column].fillna(
                    mode_value.iloc[0]
                )

        elif method == "all":

            if pd.api.types.is_numeric_dtype(df[column]):

                df[column] = df[column].fillna(
                    df[column].mean()
                )

            else:

                df[column] = df[column].fillna(
                    df[column].mode().iloc[0]
                )

    return df


# =========================================================
# FILL MISSING NUMERICAL VALUES
# =========================================================

def fill_numerical_missing_values(df):

    numerical_columns = get_numerical_columns(df)

    for column in numerical_columns:

        if df[column].isnull().sum() > 0:

            df[column] = df[column].fillna(
                df[column].mean()
            )

    return df


# =========================================================
# FILL MISSING CATEGORICAL VALUES
# =========================================================

def fill_categorical_missing_values(df):

    categorical_columns = get_categorical_columns(df)

    for column in categorical_columns:

        if df[column].isnull().sum() > 0:

            df[column] = df[column].fillna(
                df[column].mode()[0]
            )

    return df


# =========================================================
# ENCODE CATEGORICAL DATA
# =========================================================

def encode_data(df, encoding_method="label", columns=None):

    if columns is None:

        columns = get_categorical_columns(df)

    if encoding_method == "onehot":

        return pd.get_dummies(
            df,
            columns=columns,
            drop_first=False
        )

    for column in columns:

        if column in df.columns:

            df[column] = df[column].astype("category").cat.codes

    return df


# =========================================================
# APPLY SCALING
# =========================================================

def scale_data(df, scaling_method, selected_columns=None):

    if selected_columns is None:

        selected_columns = get_numerical_columns(df)

    selected_columns = [
        column for column in selected_columns
        if column in df.columns
    ]

    if not selected_columns:

        return df

    if scaling_method == "standard":

        scaler = StandardScaler()

    elif scaling_method == "minmax":

        scaler = MinMaxScaler()

    elif scaling_method == "normalize":

        scaler = Normalizer()

    else:

        return df

    values = df[selected_columns].copy()

    # Scikit-learn's Normalizer cannot process missing numeric values.
    values = values.fillna(
        values.mean()
    ).fillna(0)

    df[selected_columns] = scaler.fit_transform(
        values
    )

    return df


def prepare_missing_value_page(df, filename, form_data=None, submitted=False, processed_folder="processed"):

    missing_columns = get_missing_columns(df)
    page_data = {
        "missing_summary": get_missing_values(df).to_html(classes="table table-bordered"),
        "missing_columns": missing_columns,
        "table": None,
        "message": None,
        "code_example": (
            "# Example:\n"
            "# df = df.copy()\n"
            "# columns = ['CGPA', 'Gender']\n"
            "# for column in columns:\n"
            "#     if df[column].isnull().sum() > 0:\n"
            "#         df[column] = df[column].fillna(df[column].mean())\n"
        )
    }

    if not submitted:
        return page_data

    form_data = form_data or {}
    method = form_data.get("method", "mean")
    selected_columns = form_data.getlist("columns") or missing_columns
    df = fill_missing_values(df, method, selected_columns)

    if method == "all":
        page_data["message"] = (
            "Missing values filled with the default mean for numeric columns "
            "and mode for categorical columns."
        )
    else:
        page_data["message"] = f"Selected columns were filled using {method.upper()}."

    page_data["code_example"] = (
        "df = df.copy()\n"
        f"columns = {selected_columns}\n"
        f"method = '{method}'\n"
        "for column in columns:\n"
        "    if column in df.columns and df[column].isnull().sum() > 0:\n"
        "        if pd.api.types.is_numeric_dtype(df[column]):\n"
        "            if method == 'mean':\n"
        "                df[column] = df[column].fillna(df[column].mean())\n"
        "            elif method == 'median':\n"
        "                df[column] = df[column].fillna(df[column].median())\n"
        "        else:\n"
        "            df[column] = df[column].fillna(df[column].mode().iloc[0])\n"
    )
    df.to_csv(os.path.join(processed_folder, "missing_" + filename), index=False)
    page_data["table"] = df.head(10).to_html(classes="table table-bordered", index=False)
    return page_data


def prepare_encoding_page(df, filename, form_data=None, submitted=False, processed_folder="processed"):

    categorical_columns = get_categorical_columns(df)
    page_data = {
        "categorical_columns": categorical_columns,
        "table": None,
        "message": None,
        "code_example": None
    }

    if not submitted:
        return page_data

    form_data = form_data or {}
    encoding_method = form_data.get("method", "label")
    selected_columns = form_data.getlist("columns") or categorical_columns
    df = encode_data(df, encoding_method=encoding_method, columns=selected_columns)
    page_data["message"] = f"Categorical data encoded using {encoding_method.upper()} encoding."
    page_data["code_example"] = (
        "# Example:\n"
        f"columns = {selected_columns}\n"
        f"method = '{encoding_method}'\n"
        "if method == 'onehot':\n"
        "    df = pd.get_dummies(df, columns=columns)\n"
        "else:\n"
        "    for column in columns:\n"
        "        df[column] = df[column].astype('category').cat.codes\n"
    )
    df.to_csv(os.path.join(processed_folder, "encoded_" + filename), index=False)
    page_data["table"] = df.head(10).to_html(classes="table table-bordered", index=False)
    return page_data


def prepare_scaling_page(df, filename, form_data=None, submitted=False, processed_folder="processed"):

    numerical_columns = get_numerical_columns(df)
    page_data = {
        "numerical_columns": numerical_columns,
        "table": None,
        "method_name": None,
        "error": None
    }

    if not submitted:
        return page_data

    form_data = form_data or {}
    scaling_method = form_data.get("scaling_method", "")
    selected_columns = form_data.getlist("columns") or numerical_columns
    method_names = {
        "standard": "Standard Scaler",
        "minmax": "Min-Max Scaler",
        "normalize": "Normalization"
    }

    if scaling_method not in method_names:
        page_data["error"] = "Please select a scaling method."
        return page_data

    df = scale_data(df, scaling_method, selected_columns)
    page_data["method_name"] = method_names[scaling_method]
    df.to_csv(os.path.join(processed_folder, "scaled_" + filename), index=False)
    page_data["table"] = df.head(10).to_html(classes="table table-bordered", index=False)
    return page_data