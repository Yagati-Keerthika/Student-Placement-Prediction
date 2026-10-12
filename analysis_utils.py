def get_analysis_feature_columns(df, target_column=None):
    columns = df.columns.tolist()
    excluded = {
        column for column in columns
        if column.casefold() == "id" or column.casefold().endswith("_id")
    }

    if target_column == "placement_status":
        excluded.update(
            column for column in columns
            if column.casefold().startswith("salary_package")
        )

    return [
        column for column in columns
        if column != target_column and column not in excluded
    ]


def get_numeric_analysis_features(df, target_column=None):
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    analysis_columns = get_analysis_feature_columns(df, target_column)
    return [
        column for column in numeric_columns
        if column in analysis_columns
    ]
