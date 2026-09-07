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
- Make predictions from entered feature values

## Project Structure

- `app.py`: Flask routes and page rendering
- `preprocessing.py`: reusable preprocessing workflows
- `main.py`: general placement prediction workflow
- `linear_regression.py`: linear regression training and prediction
- `logistic_regression.py`: logistic regression training and prediction
- `decision_tree.py`: decision-tree training, metrics, visualization, and prediction
- `random_forest.py`: random-forest training, metrics, visualization, and prediction
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
