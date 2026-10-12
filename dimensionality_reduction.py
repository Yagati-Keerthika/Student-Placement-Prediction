import base64
import io

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import umap


def run_dimensionality_reduction(df, form_data=None, submitted=False):
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    default_features = [
        column for column in ["cgpa", "aptitude_score", "coding_skill_score"]
        if column in numeric_columns
    ]

    page = {
        "feature_columns": numeric_columns,
        "selected_features": default_features,
        "n_clusters": 3,
        "method": "pca",
        "metrics": None,
        "graph": None,
        "comparison_graph": None,
        "prediction_result": None,
        "error": None
    }

    if not submitted:
        return page

    features = [column for column in form_data.getlist("feature") if column in numeric_columns]
    if not features:
        features = default_features

    try:
        clusters = max(2, min(int(form_data.get("n_clusters", 3)), 10))
    except (TypeError, ValueError):
        clusters = 3

    method = form_data.get("method", "pca")
    page.update({
        "selected_features": features,
        "n_clusters": clusters,
        "method": method
    })

    if len(features) < 2:
        page["error"] = "Select at least two numeric features."
        return page

    if len(df) < clusters:
        page["error"] = "The number of clusters cannot be greater than the number of students."
        return page

    try:
        data = df[features].copy()
        data = data.fillna(data.median()).fillna(0)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # PCA transformation
        pca_model = PCA(n_components=2)
        pca_data = pca_model.fit_transform(scaled_data)

        # t-SNE transformation
        tsne_model = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        tsne_data = tsne_model.fit_transform(scaled_data)

        # UMAP transformation
        umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        umap_data = umap_model.fit_transform(scaled_data)

        # Select data based on method
        if method == "pca":
            reduction_data = pca_data
            model_name = "PCA"
            axis_labels = ("Principal Component 1", "Principal Component 2")
        elif method == "tsne":
            reduction_data = tsne_data
            model_name = "t-SNE"
            axis_labels = ("t-SNE Dimension 1", "t-SNE Dimension 2")
        else:  # umap
            reduction_data = umap_data
            model_name = "UMAP"
            axis_labels = ("UMAP Dimension 1", "UMAP Dimension 2")

        # K-Means clustering
        kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(reduction_data)

        # Calculate silhouette score
        if clusters < len(data):
            score_data = reduction_data
            score_labels = labels
            if len(data) > 2000:
                sample_indices = pd.Series(range(len(data))).sample(
                    n=2000,
                    random_state=42
                ).to_numpy()
                score_data = reduction_data[sample_indices]
                score_labels = labels[sample_indices]
            silhouette = silhouette_score(score_data, score_labels)
        else:
            silhouette = None

        # Create visualization for selected method
        plt.figure(figsize=(8, 5))
        plt.scatter(
            reduction_data[:, 0],
            reduction_data[:, 1],
            c=labels,
            cmap="viridis",
            edgecolors="black",
            alpha=0.7
        )
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])
        plt.title(f"{model_name} with K-Means Clusters")
        plt.colorbar(label="Cluster")
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        page["graph"] = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        # Create comparison graph
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        methods_data = [
            (pca_data, "PCA", "Principal Component 1", "Principal Component 2"),
            (tsne_data, "t-SNE", "t-SNE Dim 1", "t-SNE Dim 2"),
            (umap_data, "UMAP", "UMAP Dim 1", "UMAP Dim 2")
        ]

        for ax, (data_points, title, xlabel, ylabel) in zip(axes, methods_data):
            kmeans_temp = KMeans(n_clusters=clusters, random_state=42, n_init=10)
            temp_labels = kmeans_temp.fit_predict(data_points)
            scatter = ax.scatter(
                data_points[:, 0],
                data_points[:, 1],
                c=temp_labels,
                cmap="viridis",
                edgecolors="black",
                alpha=0.7
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} Clustering")
            plt.colorbar(scatter, ax=ax, label="Cluster")

        plt.tight_layout()
        buffer_comparison = io.BytesIO()
        plt.savefig(buffer_comparison, format="png")
        buffer_comparison.seek(0)
        page["comparison_graph"] = base64.b64encode(buffer_comparison.read()).decode("utf-8")
        plt.close()

        # Store metrics
        page["metrics"] = {
            "model": f"{model_name} + K-Means",
            "selected_features": features,
            "n_clusters": clusters,
            "training_samples": len(data),
            "cluster_counts": pd.Series(labels).value_counts().sort_index().tolist(),
            "labels": labels.tolist(),
            "inertia": float(round(kmeans.inertia_, 4)),
            "silhouette_score": (
                float(round(silhouette, 4))
                if silhouette is not None else None
            )
        }

        # Add PCA-specific metrics
        if method == "pca":
            page["metrics"]["n_components"] = pca_model.n_components_
            page["metrics"]["explained_variance"] = [
                float(round(value * 100, 2))
                for value in pca_model.explained_variance_ratio_
            ]
            page["metrics"]["variance_retained"] = float(round(sum(pca_model.explained_variance_ratio_) * 100, 2))

        # Cluster labels based on placement rates
        placement_rates = {}
        if "placement_status" in df.columns:
            placed = df["placement_status"].astype(str).str.lower().eq("placed")
            placement_rates = pd.Series(placed).groupby(labels).mean().to_dict()

        ordered_clusters = sorted(
            range(clusters),
            key=lambda cluster: placement_rates.get(cluster, 0)
        )
        page["cluster_labels"] = {
            cluster: [
                "Low Placement Chance",
                "Medium Placement Chance",
                "High Placement Chance"
            ][position]
            for position, cluster in enumerate(ordered_clusters)
        } if clusters == 3 else {
            cluster: f"Cluster {position + 1}"
            for position, cluster in enumerate(ordered_clusters)
        }

        # Prediction
        if form_data.get("action") == "predict":
            values = []
            for feature in features:
                raw_value = form_data.get(feature, "")
                try:
                    values.append(float(raw_value))
                except (TypeError, ValueError):
                    page["error"] = f"Please enter a valid number for {feature}."
                    return page

            prediction_data = pd.DataFrame([values], columns=features)
            prediction_data = prediction_data.fillna(data.median()).fillna(0)
            scaled_prediction = scaler.transform(prediction_data)

            if method == "pca":
                prediction_reduced = pca_model.transform(scaled_prediction)
            elif method == "tsne":
                prediction_reduced = tsne_model.transform(scaled_prediction)
            else:  # umap
                prediction_reduced = umap_model.transform(scaled_prediction)

            predicted_cluster = int(kmeans.predict(prediction_reduced)[0])
            page["prediction_result"] = page["cluster_labels"][predicted_cluster]

    except Exception as error:
        page["error"] = str(error)

    return page


def get_comparison_metrics(df, features=None):
    """
    Calculate and compare metrics for all three dimensionality reduction methods.
    """
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if features is None:
        features = [
            column for column in ["cgpa", "aptitude_score", "coding_skill_score"]
            if column in numeric_columns
        ]

    data = df[features].copy()
    data = data.fillna(data.median()).fillna(0)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Apply all three methods
    pca_model = PCA(n_components=2)
    pca_data = pca_model.fit_transform(scaled_data)

    tsne_model = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    tsne_data = tsne_model.fit_transform(scaled_data)

    umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_data = umap_model.fit_transform(scaled_data)

    # Compute metrics for each method
    metrics = {}
    for method_name, data_points in [("PCA", pca_data), ("t-SNE", tsne_data), ("UMAP", umap_data)]:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_points)
        silhouette = silhouette_score(data_points, labels)

        metrics[method_name] = {
            "inertia": float(round(kmeans.inertia_, 4)),
            "silhouette_score": float(round(silhouette, 4))
        }

        if method_name == "PCA":
            metrics[method_name]["variance_retained"] = float(
                round(sum(pca_model.explained_variance_ratio_) * 100, 2)
            )

    return metrics
