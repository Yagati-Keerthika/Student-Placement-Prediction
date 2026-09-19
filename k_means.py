import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _fit_model(df, features, clusters):
    data = df[features].copy()

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = data[column].fillna(data[column].median()).fillna(0)
        else:
            data[column] = data[column].fillna("Unknown")

    data = pd.get_dummies(data).astype(float)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    model = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_data)
    centers = pd.DataFrame(model.cluster_centers_, columns=data.columns).round(2)

    result = {
        "model": model,
        "clustered_data": data.assign(cluster=labels),
        "cluster_centres": centers,
        "labels": labels,
        "scaler": scaler,
        "columns": data.columns,
        "features": features
    }
    return result


def train_kmeans(df, form_data=None, submitted=False):
    columns = df.columns.tolist()
    default_features = [
        column for column in ["cgpa", "aptitude_score", "coding_skill_score"]
        if column in columns
    ] or columns[:1]

    page = {
        "feature_columns": columns,
        "selected_features": default_features,
        "n_clusters": 3,
        "metrics": None,
        "prediction_result": None,
        "input_options": {},
        "error": None
    }

    if not submitted:
        return page

    features = [column for column in form_data.getlist("feature") if column in columns]
    try:
        clusters = max(2, min(int(form_data.get("n_clusters", 3)), 20))
    except (TypeError, ValueError):
        clusters = 3

    page.update({"selected_features": features, "n_clusters": clusters})

    if not features:
        page["error"] = "Select at least one feature."
        return page

    try:
        result = _fit_model(df, features, clusters)
    except Exception as error:
        page["error"] = str(error)
        return page

    centers = result["cluster_centres"]
    cluster_labels = result["labels"]
    placement_rates = {}

    if "placement_status" in df.columns:
        placed = df["placement_status"].astype(str).str.lower().eq("placed")
        placement_rates = pd.Series(placed).groupby(cluster_labels).mean().to_dict()

    order = sorted(
        range(clusters),
        key=lambda cluster: placement_rates.get(cluster, 0)
    )
    chance_labels = {
        cluster: [
            "Low Placement Chance",
            "Medium Placement Chance",
            "High Placement Chance"
        ][position]
        for position, cluster in enumerate(order)
    } if clusters == 3 else {
        cluster: f"Cluster {position + 1}"
        for position, cluster in enumerate(order)
    }

    page["metrics"] = {
        "selected_features": features,
        "n_clusters": clusters,
        "cluster_counts": result["clustered_data"]["cluster"].value_counts().sort_index().tolist(),
        "training_samples": len(result["clustered_data"]),
        "placement_rates": [
            round(placement_rates.get(cluster, 0) * 100, 2)
            for cluster in range(clusters)
        ],
        "cluster_labels": chance_labels,
        "cluster_centers": centers.to_html(classes="table table-bordered", index=False)
    }
    result["cluster_labels"] = chance_labels
    page["input_options"] = {
        column: sorted(df[column].dropna().astype(str).unique().tolist())
        for column in features
        if not pd.api.types.is_numeric_dtype(df[column])
    }
    page["_result"] = result
    return page


def predict_kmeans(df, form_data, page):
    if page.get("error") or page.get("_result") is None:
        return page

    values = {}
    for feature in page["selected_features"]:
        value = form_data.get(feature, "")
        if value == "":
            page["error"] = f"Please enter a value for {feature}."
            return page

        if pd.api.types.is_numeric_dtype(df[feature]):
            try:
                value = float(value)
            except ValueError:
                page["error"] = f"{feature} must be a valid number."
                return page
        values[feature] = value

    result = page["_result"]
    row = pd.DataFrame([values])
    row = pd.get_dummies(row).reindex(
        columns=result["columns"],
        fill_value=0
    ).astype(float)
    row = result["scaler"].transform(row)
    cluster = int(result["model"].predict(row)[0])
    page["prediction_result"] = result["cluster_labels"].get(
        cluster,
        f"Cluster {cluster + 1}"
    )
    page.pop("_result", None)
    return page