# Student Placement Prediction

A Flask-based machine learning application for exploring and predicting student placement outcomes.

## Project Description

This project provides an interactive workflow for loading a student dataset, inspecting and preprocessing the data, visualizing patterns, and training machine learning models. Users can select input features, choose model settings, view evaluation metrics, and enter values for placement predictions.

## Features

- Upload and preview student placement datasets
- View missing values and dataset summaries
- Fill missing values using mean, median, mode, or combined strategies
- Encode categorical columns with label or one-hot encoding
- Scale numeric columns with standard, min-max, or normalization methods
- Explore charts and dataset visualizations
- Train and evaluate:
  - Logistic Regression
  - Linear Regression
  - Decision Tree
  - Random Forest
- Select decision-tree and random-forest settings such as criterion, number of trees, and maximum depth
- Compare PCA, t-SNE, and UMAP visualizations using the same stratified sample of up to 1,000 rows
- Compare ROC curves and AUC scores for Logistic Regression, Decision Tree, Random Forest, AdaBoost, and XGBoost using one shared stratified test split
- Exclude identifier columns and placement-derived salary fields from placement-classification features to avoid target leakage
- Make predictions from entered feature values

### PCA, t-SNE, and UMAP comparison

The comparison page follows these steps:

1. Choose a target column for point colors and at least two numeric input features.
2. Remove rows without a target; fill missing feature values with their median.
3. Use a stratified sample of at most 1,000 rows, then scale the features.
4. Run PCA, t-SNE, and UMAP on the same scaled rows to get two coordinates per row.
5. Draw one scatter plot per method, coloring points by the target class.

PCA keeps directions with high overall variance. t-SNE focuses on keeping nearby
points together, while UMAP also aims to preserve local neighborhoods. Their
coordinates are for visualization, not a direct measure of model accuracy.

## Project Structure

- `app.py`: Flask routes and page rendering
- `preprocessing.py`: reusable preprocessing workflows
- `main.py`: general placement prediction workflow
- `linear_regression.py`: linear regression training and prediction
- `logistic_regression.py`: logistic regression training and prediction
- `decision_tree.py`: decision-tree training, metrics, visualization, and prediction
- `random_forest.py`: random-forest training, metrics, visualization, and prediction
- `dimensionality_comparison.py`: PCA, t-SNE, and UMAP visualization comparison
- `roc_comparison.py`: shared-split ROC/AUC classifier comparison
- `templates/`: HTML pages
- `static/`: CSS and chart assets
- `processed/`: generated processed datasets
- `uploads/`: uploaded datasets

## Running the Application

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the Flask application:

```bash
python app.py
```

Open the local URL shown by Flask in a browser.

## Dataset

The project includes `student_placement_prediction_dataset_2026.csv`, containing student academic, technical, communication, attendance, and placement-related features.
