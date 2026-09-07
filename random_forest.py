import base64
import io

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree


def build_random_forest_model(df, feature_columns, target_column, n_estimators=30, max_depth=12, random_state=42):

	numeric_columns = df.select_dtypes(
		include=[np.number]
	).columns.tolist()

	selected_features = [
		column for column in feature_columns
		if column in numeric_columns and column != target_column
	]

	if not selected_features:
		return None, {
			"error": "Please choose at least one valid numeric input feature."
		}, None

	if target_column not in df.columns:
		return None, {
			"error": "Selected target column does not exist."
		}, None

	model_df = df[selected_features + [target_column]].dropna().reset_index(drop=True)

	if model_df.empty:
		return None, {
			"error": "No valid rows are available after removing missing values."
		}, None

	X = model_df[selected_features]
	y = model_df[target_column]

	if y.nunique() != 2:
		return None, {
			"error": "The target feature must contain exactly two classes."
		}, None

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

	model = RandomForestClassifier(
		n_estimators=n_estimators,
		max_depth=max_depth,
		n_jobs=-1,
		random_state=random_state
	)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	cm = confusion_matrix(y_test, y_pred, labels=list(range(len(target_labels))))

	forest_depth = max(
		[tree.get_depth() for tree in model.estimators_]
	) if hasattr(model, "estimators_") and len(model.estimators_) > 0 else None

	first_tree = model.estimators_[0]
	fig, ax = plt.subplots(figsize=(18, 9))
	plot_tree(
		first_tree,
		feature_names=selected_features,
		class_names=[str(label) for label in target_labels],
		filled=True,
		rounded=True,
		max_depth=5,
		ax=ax
	)
	ax.set_title(
		f"Representative Tree 1 of {n_estimators} | depth limit: {max_depth} | displayed depth: 5"
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
		"n_estimators": n_estimators,
		"max_depth_limit": max_depth,
		"max_depth": forest_depth,
		"feature_importances": [float(value) for value in model.feature_importances_],
		"tree_image": f"data:image/png;base64,{encoded_image}",
		"tree_count_explanation": f"This forest uses {n_estimators} decision trees and combines their predictions.",
		"train_samples": len(X_train),
		"test_samples": len(X_test),
		"selected_features": selected_features,
		"target_column": target_column,
		"target_labels": target_labels
	}

	return model, metrics, target_labels


def predict_random_forest(model, prediction_values, feature_columns, target_labels):

	prediction_row = pd.DataFrame(
		[prediction_values],
		columns=feature_columns
	)

	prediction_code = int(model.predict(prediction_row)[0])
	prediction_result = target_labels[prediction_code]
	prediction_probability = float(
		model.predict_proba(prediction_row).max()
	)

	return prediction_result, prediction_probability


def prepare_random_forest_page(df, form_data=None, submitted=False):

	numeric_columns = df.select_dtypes(
		include=[np.number]
	).columns.tolist()
	target_candidates = [
		column for column in df.columns
		if df[column].nunique(dropna=True) == 2
	]
	default_target = (
		"placement_status"
		if "placement_status" in target_candidates
		else target_candidates[0] if target_candidates else None
	)
	default_features = [
		column for column in ["cgpa", "aptitude_score", "coding_skill_score"]
		if column in numeric_columns
	] or numeric_columns[:1]
	max_depth_options = [5, 8, 12, 15]

	page_data = {
		"numeric_columns": numeric_columns,
		"target_candidates": target_candidates,
		"selected_features": default_features,
		"target_column": default_target,
		"n_estimators": 30,
		"max_depth": 12,
		"max_depth_options": max_depth_options,
		"metrics": None,
		"prediction_result": None,
		"prediction_probability": None,
		"error": None
	}

	if not numeric_columns or not target_candidates:
		page_data["error"] = (
			"This dataset does not contain numeric features and a binary target."
		)
		return page_data

	if not submitted:
		return page_data

	form_data = form_data or {}
	selected_features = form_data.getlist("feature")
	target_column = form_data.get("target_feature", default_target)

	try:
		n_estimators = int(form_data.get("n_estimators", 30))
	except (TypeError, ValueError):
		n_estimators = 30

	n_estimators = max(10, min(n_estimators, 100))

	try:
		max_depth = int(form_data.get("max_depth", 12))
	except (TypeError, ValueError):
		max_depth = 12

	page_data.update({
		"selected_features": selected_features,
		"target_column": target_column,
		"n_estimators": n_estimators,
		"max_depth": max_depth
	})

	if not selected_features:
		page_data["error"] = "Please select at least one input feature."
		return page_data

	if target_column not in target_candidates:
		page_data["error"] = "Please select a valid binary target feature."
		return page_data

	if max_depth not in max_depth_options:
		page_data["error"] = "Please select a valid forest depth."
		return page_data

	model, metrics, target_labels = build_random_forest_model(
		df,
		selected_features,
		target_column,
		n_estimators=n_estimators,
		max_depth=max_depth
	)

	if model is None:
		page_data["error"] = metrics.get("error")
		return page_data

	page_data["metrics"] = metrics
	prediction_values = {}

	for feature in selected_features:
		value = form_data.get(feature, "")

		if value == "":
			page_data["error"] = f"Please enter a value for {feature}."
			return page_data

		try:
			prediction_values[feature] = float(value)
		except (TypeError, ValueError):
			page_data["error"] = f"{feature} must be a valid number."
			return page_data

	(
		page_data["prediction_result"],
		page_data["prediction_probability"]
	) = predict_random_forest(
		model,
		prediction_values,
		selected_features,
		target_labels
	)

	return page_data
