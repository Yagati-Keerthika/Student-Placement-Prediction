import base64
import io

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# This model is simple: training data learns patterns, and then it predicts placement for a new student.
def build_decision_tree_model(df, feature_columns, target_column, criterion="entropy", max_depth=10, random_state=42):

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = [
        column for column in feature_columns
        if column in numeric_columns and column != target_column
    ]

    if not selected_features:
        return None, {"error": "Please choose at least one valid numeric input feature."}, None

    if target_column not in df.columns:
        return None, {"error": "Selected target column does not exist."}, None

    # Keep only useful columns and remove missing values
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
        random_state=random_state,
        stratify=y
    )

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(target_labels))))

    # Show one tree graph to explain the decision process
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(
        model,
        feature_names=selected_features,
        class_names=[str(label) for label in target_labels],
        filled=True,
        rounded=True,
        max_depth=5,
        ax=ax
    )
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close(fig)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "criterion": criterion,
        "max_depth": model.get_depth(),
        "feature_importances": [float(value) for value in model.feature_importances_],
        "tree_image": f"data:image/png;base64,{encoded_image}",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "selected_features": selected_features,
        "target_column": target_column,
        "target_labels": target_labels
    }

    return model, metrics, target_labels


def predict_decision_tree(model, prediction_values, feature_columns, target_labels):
    prediction_row = pd.DataFrame([prediction_values], columns=feature_columns)
    prediction_code = int(model.predict(prediction_row)[0])
    prediction_result = target_labels[prediction_code]
    prediction_probability = float(model.predict_proba(prediction_row).max())
    return prediction_result, prediction_probability

