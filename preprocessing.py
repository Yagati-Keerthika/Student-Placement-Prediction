import pandas as pd
import numpy as np

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

    df[selected_columns] = scaler.fit_transform(
        df[selected_columns]
    )

    return df