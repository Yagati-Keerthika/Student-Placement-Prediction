import base64
import io

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def run_pca_kmeans(df, form_data=None, submitted=False):
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    default_features = [
        column for column in ["cgpa", "aptitude_score", "coding_skill_score"]
        if column in numeric_columns
    ]

    page = {
        "feature_columns": numeric_columns,
        "selected_features": default_features,
        "n_clusters": 3,
        "metrics": None,
        "graph": None,
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

    page.update({
        "selected_features": features,
        "n_clusters": clusters
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

        pca_model = PCA(n_components=2)
        pca_data = pca_model.fit_transform(scaled_data)

        kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pca_data)
        if clusters < len(data):
            score_data = pca_data
            score_labels = labels
            if len(data) > 2000:
                sample_indices = pd.Series(range(len(data))).sample(
                    n=2000,
                    random_state=42
                ).to_numpy()
                score_data = pca_data[sample_indices]
                score_labels = labels[sample_indices]
            silhouette = silhouette_score(score_data, score_labels)
        else:
            silhouette = None

        plt.figure(figsize=(8, 5))
        plt.scatter(
            pca_data[:, 0],
            pca_data[:, 1],
            c=labels,
            cmap="viridis",
            edgecolors="black"
        )
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA with K-Means Clusters")
        plt.colorbar(label="Cluster")
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        page["graph"] = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        page["metrics"] = {
            "model": "PCA + K-Means",
            "selected_features": features,
            "n_clusters": clusters,
            "n_components": pca_model.n_components_,
            "training_samples": len(data),
            "cluster_counts": pd.Series(labels).value_counts().sort_index().tolist(),
            "labels": labels.tolist(),
            "inertia": float(round(kmeans.inertia_, 4)),
            "explained_variance": [
                float(round(value * 100, 2))
                for value in pca_model.explained_variance_ratio_
            ],
            "silhouette_score": (
                float(round(silhouette, 4))
                if silhouette is not None else None
            ),
            "variance_retained": float(round(sum(pca_model.explained_variance_ratio_) * 100, 2))
        }

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
            prediction_pca = pca_model.transform(scaler.transform(prediction_data))
            predicted_cluster = int(kmeans.predict(prediction_pca)[0])
            page["prediction_result"] = page["cluster_labels"][predicted_cluster]
    except Exception as error:
        page["error"] = str(error)

    return page