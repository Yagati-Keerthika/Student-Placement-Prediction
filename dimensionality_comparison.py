import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import umap

from analysis_utils import get_numeric_analysis_features


MAX_EMBEDDING_SAMPLES = 1000


def run_dimensionality_comparison(df, form_data=None, submitted=False):
    """Prepare the page, then compare PCA, t-SNE, and UMAP when submitted."""
    target_candidates = [
        column for column in df.columns
        if 2 <= df[column].nunique(dropna=True) <= 20
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
        "max_samples": MAX_EMBEDDING_SAMPLES,
        "metrics": None,
        "graph": None,
        "error": None
    }

    if not submitted:
        return page

    # Keep only valid numeric inputs and check the user's choices.
    selected_features = [
        column for column in selected_features
        if column in feature_columns
    ]
    page["selected_features"] = selected_features

    if len(selected_features) < 2:
        page["error"] = "Select at least two numeric input features."
        return page
    if target_column not in target_candidates:
        page["error"] = "Select a valid target feature with 2 to 20 classes."
        return page

    model_df = df[selected_features + [target_column]].dropna(
        subset=[target_column]
    )
    if model_df.empty:
        page["error"] = "No valid rows are available for the selected target."
        return page

    labels = model_df[target_column].astype("category")
    class_labels = labels.cat.categories.tolist()
    if len(class_labels) < 2:
        page["error"] = "The target feature must contain at least two classes."
        return page

    if len(model_df) > MAX_EMBEDDING_SAMPLES:
        try:
            sample_indices, _ = train_test_split(
                np.arange(len(model_df)),
                train_size=MAX_EMBEDDING_SAMPLES,
                random_state=42,
                stratify=labels.cat.codes
            )
        except ValueError as error:
            page["error"] = (
                f"Could not create a stratified {MAX_EMBEDDING_SAMPLES}-row sample: "
                f"{error}"
            )
            return page
        model_df = model_df.iloc[sample_indices]
        labels = labels.iloc[sample_indices]

    if len(model_df) < 4:
        page["error"] = "At least four valid rows are needed for the comparison."
        return page

    # Fill missing inputs, then scale every feature to a comparable range.
    feature_data = model_df[selected_features].copy()
    feature_data = feature_data.fillna(feature_data.median()).fillna(0)
    scaled_data = StandardScaler().fit_transform(feature_data)
    label_codes = labels.cat.codes.to_numpy()

    # Each algorithm converts the same scaled rows into two plot coordinates.
    pca_model = PCA(n_components=2)
    pca_data = pca_model.fit_transform(scaled_data)

    row_count = len(model_df)
    tsne_perplexity = min(30.0, max(1.0, (row_count - 1) / 3.0))
    tsne_data = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        random_state=42,
        max_iter=500,
        init="pca",
        learning_rate="auto"
    ).fit_transform(scaled_data)

    umap_neighbors = min(15, row_count - 1)
    umap_data = umap.UMAP(
        n_components=2,
        n_neighbors=umap_neighbors,
        min_dist=0.1,
        n_epochs=100,
        random_state=42
    ).fit_transform(scaled_data)

    # Draw one scatter plot per algorithm, using the target class for color.
    methods = [
        (
            pca_data,
            "PCA",
            "Principal component 1",
            "Principal component 2"
        ),
        (tsne_data, "t-SNE", "t-SNE dimension 1", "t-SNE dimension 2"),
        (umap_data, "UMAP", "UMAP dimension 1", "UMAP dimension 2")
    ]
    colors = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for axis, (coordinates, title, xlabel, ylabel) in zip(axes, methods):
        for class_index, label in enumerate(class_labels):
            class_mask = label_codes == class_index
            axis.scatter(
                coordinates[class_mask, 0],
                coordinates[class_mask, 1],
                s=10,
                alpha=0.65,
                color=colors(class_index),
                label=str(label),
                rasterized=True
            )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.legend(title=target_column, fontsize="small")

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    page["graph"] = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Return the chart and a few details for the page.
    page["metrics"] = {
        "features": selected_features,
        "target_column": target_column,
        "samples": len(model_df),
        "original_samples": int(df[target_column].notna().sum()),
        "pca_variance_retained": float(
            round(pca_model.explained_variance_ratio_.sum() * 100, 2)
        )
    }
    return page
