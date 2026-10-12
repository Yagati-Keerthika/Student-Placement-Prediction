import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from analysis_utils import get_numeric_analysis_features


MODEL_OPTIONS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest"
}


def prepare_k_fold_page(df, form_data=None, submitted=False):
    target_candidates = df.columns.tolist()
    target_column = (
        form_data.get("target_feature")
        if form_data is not None
        else None
    )
    if target_column not in target_candidates:
        target_column = (
            "placement_status"
            if "placement_status" in target_candidates
            else (target_candidates[-1] if target_candidates else None)
        )

    numeric_columns = get_numeric_analysis_features(df, target_column)
    selected_features = (
        form_data.getlist("feature")
        if form_data is not None
        else []
    )
    if not submitted and not selected_features:
        selected_features = [
            column for column in numeric_columns
            if column != target_column
        ]

    try:
        folds = int(form_data.get("folds", 5)) if form_data is not None else 5
    except (TypeError, ValueError):
        folds = 5
    folds = max(2, min(folds, 10))

    model_key = (
        form_data.get("model", "logistic_regression")
        if form_data is not None
        else "logistic_regression"
    )
    if model_key not in MODEL_OPTIONS:
        model_key = "logistic_regression"

    page = {
        "numeric_columns": numeric_columns,
        "target_candidates": target_candidates,
        "selected_features": selected_features,
        "target_column": target_column,
        "folds": folds,
        "model_options": MODEL_OPTIONS,
        "selected_model": model_key,
        "metrics": None,
        "error": None
    }

    if not submitted:
        return page

    features = [
        column for column in selected_features
        if column in numeric_columns and column != target_column
    ]
    page["selected_features"] = features

    if not features:
        page["error"] = "Select at least one numeric input feature."
        return page
    if target_column not in df.columns:
        page["error"] = "Select a valid target feature."
        return page

    model_df = df[features + [target_column]].dropna().reset_index(drop=True)
    if model_df.empty:
        page["error"] = "No valid rows are available after removing missing values."
        return page

    target_values = model_df[target_column].astype("category")
    class_counts = target_values.value_counts()
    if len(class_counts) < 2:
        page["error"] = "The target feature must contain at least two classes."
        return page
    if folds > len(model_df):
        page["error"] = (
            f"The dataset has only {len(model_df)} valid rows; "
            "choose fewer folds."
        )
        return page
    if class_counts.min() < folds:
        page["error"] = (
            f"Each target class needs at least {folds} rows for "
            "stratified cross-validation."
        )
        return page

    X = model_df[features]
    y = target_values.cat.codes
    estimators = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted"
    }

    try:
        scores = cross_validate(
            estimators[model_key],
            X,
            y,
            cv=cv,
            scoring=scoring,
            error_score="raise"
        )
    except ValueError as error:
        page["error"] = str(error)
        return page

    metric_names = ("accuracy", "precision", "recall", "f1")
    fold_results = [
        {
            "fold": fold_number,
            **{
                metric: round(float(scores[f"test_{metric}"][fold_index]) * 100, 2)
                for metric in metric_names
            }
        }
        for fold_index, fold_number in enumerate(range(1, folds + 1))
    ]
    averages = {
        metric: round(float(np.mean(scores[f"test_{metric}"])) * 100, 2)
        for metric in metric_names
    }
    deviations = {
        metric: round(float(np.std(scores[f"test_{metric}"])) * 100, 2)
        for metric in metric_names
    }

    page["metrics"] = {
        "model_name": MODEL_OPTIONS[model_key],
        "folds": folds,
        "samples": len(model_df),
        "fold_results": fold_results,
        "averages": averages,
        "standard_deviations": deviations
    }
    return page
