import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from analysis_utils import get_numeric_analysis_features


def compare_roc_curves(df, form_data=None, submitted=False):
    target_candidates = [
        column for column in df.columns
        if df[column].nunique(dropna=True) == 2
    ]
    target_column = (
        form_data.get("target_feature")
        if form_data is not None
        else None
    )
    if target_column not in target_candidates:
        target_column = (
            "placement_status"
            if "placement_status" in target_candidates
            else (target_candidates[0] if target_candidates else None)
        )

    feature_columns = get_numeric_analysis_features(df, target_column)
    selected_features = (
        form_data.getlist("feature")
        if form_data is not None
        else []
    )
    if not submitted and not selected_features:
        selected_features = feature_columns[:]

    page = {
        "feature_columns": feature_columns,
        "selected_features": selected_features,
        "target_candidates": target_candidates,
        "target_column": target_column,
        "metrics": None,
        "graph": None,
        "error": None
    }

    if not submitted:
        return page

    selected_features = [
        column for column in selected_features
        if column in feature_columns
    ]
    page["selected_features"] = selected_features

    if not selected_features:
        page["error"] = "Select at least one numeric input feature."
        return page
    if target_column not in target_candidates:
        page["error"] = "Select a valid binary target feature."
        return page

    model_df = df[selected_features + [target_column]].dropna(
        subset=[target_column]
    )
    if model_df.empty:
        page["error"] = "No valid rows are available for the selected target."
        return page

    target_values = model_df[target_column].astype("category")
    target_labels = target_values.cat.categories.tolist()
    y = target_values.cat.codes.to_numpy()
    class_counts = np.bincount(y, minlength=2)
    if len(target_labels) != 2:
        page["error"] = "The target feature must contain exactly two classes."
        return page
    if class_counts.min() < 2:
        page["error"] = "Each target class needs at least two valid rows."
        return page

    if target_column == "placement_status":
        positive_index = next(
            (
                index for index, label in enumerate(target_labels)
                if str(label).strip().casefold() == "placed"
            ),
            1
        )
    else:
        positive_index = 1
    y_binary = (y == positive_index).astype(int)

    X = model_df[selected_features]
    all_missing_features = X.columns[X.isna().all()].tolist()
    if all_missing_features:
        page["error"] = (
            "These selected features contain only missing values: "
            + ", ".join(all_missing_features)
        )
        return page

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_binary,
            test_size=0.2,
            random_state=42,
            stratify=y_binary
        )
    except ValueError as error:
        page["error"] = str(error)
        return page

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    try:
        from xgboost import XGBClassifier
    except ImportError:
        page["error"] = (
            "XGBoost is required for the full classifier comparison. "
            "Install the project dependencies with pip install -r requirements.txt."
        )
        return page

    models = [
        (
            "Logistic Regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=42)
            )
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            )
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1,
                random_state=42
            )
        ),
        (
            "AdaBoost",
            AdaBoostClassifier(
                n_estimators=100,
                random_state=42
            )
        ),
        (
            "XGBoost",
            XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=42
            )
        )
    ]

    y_scores = []
    for model_name, model in models:
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)
        positive_column = list(model.classes_).index(1)
        scores = probabilities[:, positive_column]
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, scores)
        y_scores.append({
            "name": model_name,
            "fpr": false_positive_rate,
            "tpr": true_positive_rate,
            "auc": float(auc(false_positive_rate, true_positive_rate))
        })

    fig, axis = plt.subplots(figsize=(8, 6))
    for result in y_scores:
        axis.plot(
            result["fpr"],
            result["tpr"],
            label=f'{result["name"]} (AUC = {result["auc"]:.3f})'
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve Comparison")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    page["graph"] = base64.b64encode(buffer.getvalue()).decode("utf-8")
    page["metrics"] = {
        "target_column": target_column,
        "positive_label": str(target_labels[positive_index]),
        "negative_label": str(target_labels[1 - positive_index]),
        "features": selected_features,
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "results": sorted(y_scores, key=lambda result: result["auc"], reverse=True)
    }
    return page
