import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def build_adaboost_model(df, feature_columns, target_column, n_estimators=50, learning_rate=1.0):

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
	target_values = model_df[target_column].astype("category")
	y = target_values.cat.codes
	target_labels = list(target_values.cat.categories)

	if len(target_labels) != 2:
		return None, {
			"error": "The target feature must contain exactly two classes."
		}, None

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y
	)

	base_model = DecisionTreeClassifier(
		max_depth=1,
		random_state=42
	)

	model = AdaBoostClassifier(
		estimator=base_model,
		n_estimators=n_estimators,
		learning_rate=learning_rate,
		random_state=42
	)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	metrics = {
		"accuracy": accuracy_score(y_test, y_pred),
		"precision": precision_score(y_test, y_pred, zero_division=0),
		"recall": recall_score(y_test, y_pred, zero_division=0),
		"f1": f1_score(y_test, y_pred, zero_division=0),
		"confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
		"n_estimators": n_estimators,
		"learning_rate": learning_rate,
		"feature_importances": [float(value) for value in model.feature_importances_],
		"train_samples": len(X_train),
		"test_samples": len(X_test),
		"selected_features": selected_features,
		"target_column": target_column,
		"target_labels": target_labels
	}

	return model, metrics, target_labels


def predict_adaboost(model, prediction_values, feature_columns, target_labels):

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


def build_xgboost_model(df, feature_columns, target_column, n_estimators=50, learning_rate=1.0):

	try:
		from xgboost import XGBClassifier
	except ImportError:
		return None, {
			"error": "XGBoost is not installed. Run: pip install xgboost"
		}, None

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

	model_df = df[selected_features + [target_column]].dropna().reset_index(drop=True)
	target_values = model_df[target_column].astype("category")
	y = target_values.cat.codes
	target_labels = list(target_values.cat.categories)

	if len(target_labels) != 2:
		return None, {
			"error": "The target feature must contain exactly two classes."
		}, None

	X_train, X_test, y_train, y_test = train_test_split(
		model_df[selected_features],
		y,
		test_size=0.2,
		random_state=42,
		stratify=y
	)

	model = XGBClassifier(
		n_estimators=n_estimators,
		max_depth=3,
		learning_rate=learning_rate,
		subsample=0.8,
		colsample_bytree=0.8,
		eval_metric="logloss",
		random_state=42,
		n_jobs=-1
	)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	metrics = {
		"accuracy": accuracy_score(y_test, y_pred),
		"precision": precision_score(y_test, y_pred, zero_division=0),
		"recall": recall_score(y_test, y_pred, zero_division=0),
		"f1": f1_score(y_test, y_pred, zero_division=0),
		"confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
		"n_estimators": n_estimators,
		"learning_rate": learning_rate,
		"feature_importances": [float(value) for value in model.feature_importances_],
		"train_samples": len(X_train),
		"test_samples": len(X_test),
		"selected_features": selected_features,
		"target_column": target_column,
		"target_labels": target_labels
	}

	return model, metrics, target_labels

