from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session
)

import os
import io
import base64

from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np


# Machine Learning
from main import prepare_prediction_page
from linear_regression import build_linear_regression_model, predict_linear_regression
from logistic_regression import build_logistic_regression_model, predict_logistic_regression
from decision_tree import build_decision_tree_model, predict_decision_tree
from random_forest import build_random_forest_model, predict_random_forest
from boosting import (
    build_adaboost_model,
    build_xgboost_model,
    predict_adaboost
)
from k_means import predict_kmeans, train_kmeans
from k_fold import prepare_k_fold_page
from pca import run_pca_kmeans
from dimensionality_comparison import run_dimensionality_comparison
from roc_comparison import compare_roc_curves
from analysis_utils import get_numeric_analysis_features


# Preprocessing
from preprocessing import (
    load_dataset,
    get_numerical_columns,
    get_categorical_columns,
    get_missing_values,
    get_missing_columns,
    fill_numerical_missing_values,
    fill_categorical_missing_values,
    fill_missing_values,
    encode_data,
    scale_data,
    prepare_missing_value_page,
    prepare_encoding_page,
    prepare_scaling_page
)


# Configuration
from config import (
    UPLOAD_FOLDER,
    PROCESSED_FOLDER,
    SECRET_KEY
)


# Visualization
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)

app.secret_key = SECRET_KEY

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# CREATE FOLDERS
# =========================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    PROCESSED_FOLDER,
    exist_ok=True
)


def _model_form_data(df, form_data):
    target_candidates = list(df.columns)
    target_column = form_data.get("target_feature")

    if target_column not in target_candidates:
        target_column = "placement_status" if "placement_status" in target_candidates else target_candidates[-1]

    numeric_columns = get_numeric_analysis_features(df, target_column)
    selected_features = [
        column for column in form_data.getlist("feature")
        if column in numeric_columns
    ]
    if not selected_features:
        selected_features = [column for column in numeric_columns if column != target_column]

    return numeric_columns, target_candidates, selected_features, target_column


def _prediction_values(form_data, selected_features):
    if not selected_features or any(form_data.get(column) is None for column in selected_features):
        return None

    try:
        return [float(form_data.get(column)) for column in selected_features]
    except (TypeError, ValueError):
        return None


def prepare_linear_regression_page(df, form_data, submitted=False):
    numeric_columns, _, selected_features, target_column = _model_form_data(df, form_data)
    page_data = {
        "numeric_columns": numeric_columns,
        "selected_features": selected_features,
        "target_column": target_column,
        "metrics": None,
        "prediction_result": None,
        "error": None
    }

    if submitted:
        model, metrics = build_linear_regression_model(df, selected_features, target_column)
        if metrics.get("error"):
            page_data["error"] = metrics["error"]
        else:
            page_data["metrics"] = metrics
            values = _prediction_values(form_data, selected_features)
            if values is not None:
                page_data["prediction_result"] = predict_linear_regression(model, values, selected_features)

    return page_data


def _prepare_classifier_page(df, form_data, submitted, build_model, predict_model, options=None):
    numeric_columns, target_candidates, selected_features, target_column = _model_form_data(df, form_data)
    page_data = {
        "numeric_columns": numeric_columns,
        "target_candidates": target_candidates,
        "selected_features": selected_features,
        "target_column": target_column,
        "metrics": None,
        "prediction_result": None,
        "prediction_probability": None,
        "error": None
    }
    if options:
        page_data.update(options)

    if submitted:
        model_result = build_model(df, selected_features, target_column)
        model, metrics, target_labels = model_result
        if metrics.get("error"):
            page_data["error"] = metrics["error"]
        else:
            page_data["metrics"] = metrics
            values = _prediction_values(form_data, selected_features)
            if values is not None:
                result, probability = predict_model(model, values, selected_features, target_labels)
                page_data["prediction_result"] = result
                page_data["prediction_probability"] = probability

    return page_data


def prepare_logistic_regression_page(df, form_data, submitted=False):
    return _prepare_classifier_page(
        df, form_data, submitted,
        build_logistic_regression_model,
        predict_logistic_regression
    )


def prepare_decision_tree_page(df, form_data, submitted=False):
    criterion = form_data.get("criterion", "entropy")
    max_depth = int(form_data.get("max_depth", 10))
    page_data = _prepare_classifier_page(
        df, form_data, submitted,
        lambda data, features, target: build_decision_tree_model(
            data, features, target, criterion=criterion, max_depth=max_depth
        ),
        predict_decision_tree,
        {"criterion_options": ["gini", "entropy", "log_loss"], "max_depth_options": [3, 5, 10, 15],
         "criterion": criterion, "max_depth": max_depth}
    )
    return page_data


def prepare_random_forest_page(df, form_data, submitted=False):
    n_estimators = int(form_data.get("n_estimators", 30))
    max_depth = int(form_data.get("max_depth", 12))
    return _prepare_classifier_page(
        df, form_data, submitted,
        lambda data, features, target: build_random_forest_model(
            data, features, target, n_estimators=n_estimators, max_depth=max_depth
        ),
        predict_random_forest,
        {"n_estimators": n_estimators, "max_depth": max_depth, "max_depth_options": [5, 8, 12, 16]}
    )


def prepare_boosting_page(df, form_data, submitted=False):
    selected_model = form_data.get("model", "AdaBoost")
    n_estimators = int(form_data.get("n_estimators", 50))
    learning_rate = float(form_data.get("learning_rate", 1.0))
    builder = build_xgboost_model if selected_model == "XGBoost" else build_adaboost_model
    predictor = predict_adaboost
    page_data = _prepare_classifier_page(
        df, form_data, submitted,
        lambda data, features, target: builder(
            data, features, target, n_estimators=n_estimators, learning_rate=learning_rate
        ),
        predictor,
        {"model_options": ["AdaBoost", "XGBoost"], "selected_model": selected_model,
         "n_estimators": n_estimators, "learning_rate": learning_rate}
    )
    if page_data["metrics"]:
        page_data["metrics"]["model_name"] = selected_model
    return page_data


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# UPLOAD PAGE
# =========================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)

def upload():

    if request.method == "POST":

        file = request.files["file"]


        # Check file selected

        if file.filename == "":

            return "NO File Selected"


        # Secure filename

        filename = secure_filename(
            file.filename
        )


        # File path

        file_path = os.path.join(

            UPLOAD_FOLDER,

            filename
        )


        # Save file

        file.save(
            file_path
        )


        # Store filename in session

        session["dataset"] = filename


        # Go to summary

        return redirect(
            "/summary"
        )


    return render_template(
        "upload.html"
    )


# =========================================================
# SUMMARY PAGE
# =========================================================

@app.route("/summary")
def summary():

    # Get uploaded filename

    filename = session.get(
        "dataset"
    )


    # If dataset is not uploaded

    if filename is None:

        return redirect(
            "/upload"
        )


    # File path

    filepath = os.path.join(

        UPLOAD_FOLDER,

        filename
    )


    # Read CSV

    df = pd.read_csv(
        filepath
    )


    # =====================================================
    # NUMBER OF ROWS
    # =====================================================

    rows = df.shape[0]


    # =====================================================
    # NUMBER OF COLUMNS
    # =====================================================

    columns = df.shape[1]

    missing_summary = get_missing_values(df).to_html(
        classes="table table-bordered"
    )

    missing_columns = get_missing_values(df).index.tolist()


    # =====================================================
    # FIRST 10 ROWS
    # =====================================================

    preview = df.head(
        10
    ).to_html(

        classes="table table-bordered",

        index=False
    )


    # =====================================================
    # LAST 10 ROWS
    # =====================================================

    tail = df.tail(
        10
    ).to_html(

        classes="table table-bordered",

        index=False
    )


    # =====================================================
    # MISSING VALUES
    # =====================================================

    missing = get_missing_values(
        df
    ).to_html(

        classes="table table-bordered"
    )


    # =====================================================
    # DATA TYPES AND MEMORY
    # =====================================================

    buffer = io.StringIO()

    df.info(
        buf=buffer
    )

    info = buffer.getvalue()


    # =====================================================
    # STATISTICAL INFORMATION
    # =====================================================

    describe = df.describe(
        include="all"
    ).to_html(

        classes="table table-bordered"
    )


    # =====================================================
    # NUMERICAL COLUMNS
    # =====================================================

    numerical_columns = get_numerical_columns(
        df
    )


    # =====================================================
    # CATEGORICAL COLUMNS
    # =====================================================

    categorical_columns = get_categorical_columns(
        df
    )


    # =====================================================
    # SEND DATA TO SUMMARY.HTML
    # =====================================================

    return render_template(

        "summary.html",

        filename=filename,

        rows=rows,

        columns=columns,

        preview=preview,

        tail=tail,

        missing=missing,

        missing_summary=missing_summary,

        missing_columns=missing_columns,

        info=info,

        describe=describe,

        numerical_columns=numerical_columns,

        categorical_columns=categorical_columns
    )


# =========================================================
# MISSING VALUE PAGE
# =========================================================

@app.route(
    "/missing-value",
    methods=["GET", "POST"]
)

def missing_value():

    filename = session.get(
        "dataset"
    )


    if filename is None:

        return redirect(
            "/upload"
        )


    filepath = os.path.join(

        UPLOAD_FOLDER,

        filename
    )


    df = pd.read_csv(
        filepath
    )


    page_data = prepare_missing_value_page(
        df,
        filename,
        request.form,
        request.method == "POST",
        PROCESSED_FOLDER
    )

    return render_template(
        "missing_value.html",
        filename=filename,
        **page_data
    )


# =========================================================
# ENCODING PAGE
# =========================================================

@app.route(
    "/encoding",
    methods=["GET", "POST"]
)

def encoding():

    filename = session.get(
        "dataset"
    )


    if filename is None:

        return redirect(
            "/upload"
        )


    filepath = os.path.join(

        UPLOAD_FOLDER,

        filename
    )


    df = pd.read_csv(
        filepath
    )


    page_data = prepare_encoding_page(
        df,
        filename,
        request.form,
        request.method == "POST",
        PROCESSED_FOLDER
    )

    return render_template(
        "encoding.html",
        filename=filename,
        **page_data
    )


# =========================================================
# SCALING PAGE
# =========================================================

@app.route(
    "/scaling",
    methods=["GET", "POST"]
)

def scaling():

    filename = session.get(
        "dataset"
    )

    if filename is None:

        return redirect(
            "/upload"
        )

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    df = pd.read_csv(
        filepath
    )

    page_data = prepare_scaling_page(
        df,
        filename,
        request.form,
        request.method == "POST",
        PROCESSED_FOLDER
    )

    if page_data.get("error"):
        return page_data["error"]

    return render_template(
        "scaling.html",
        filename=filename,
        **page_data
    )


# =========================================================
# VISUALIZATION PAGE
# =========================================================

@app.route("/eda", methods=["GET", "POST"])
def eda():

    filename = session.get(
        "dataset"
    )


    if filename is None:

        return redirect(
            "/upload"
        )


    filepath = os.path.join(

        UPLOAD_FOLDER,

        filename
    )


    df = pd.read_csv(
        filepath
    )

    columns = list(df.columns)
    numerical_columns = get_numerical_columns(df)

    graph = None
    graph_title = None

    if request.method == "POST":

        visualization = request.form.get("visualization")
        x_column = request.form.get("x_column")
        y_column = request.form.get("y_column")

        if visualization in ["bar", "histogram", "pie", "box"] and x_column:

            plt.figure(figsize=(7, 4))

            if visualization == "bar":

                if x_column in df.columns:

                    df[x_column].value_counts().plot(kind="bar")
                    graph_title = f"{x_column} Distribution"
                    plt.xlabel(x_column)
                    plt.ylabel("Count")

            elif visualization == "histogram":

                if x_column in df.columns:

                    plt.hist(df[x_column].dropna(), bins=10, edgecolor="black")
                    graph_title = f"Histogram of {x_column}"
                    plt.xlabel(x_column)
                    plt.ylabel("Frequency")

            elif visualization == "pie":

                if x_column in df.columns:

                    counts = df[x_column].dropna().value_counts()
                    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
                    graph_title = f"{x_column} Pie Chart"

            elif visualization == "box":

                if x_column in df.columns and pd.api.types.is_numeric_dtype(df[x_column]):

                    plt.boxplot(df[x_column].dropna())
                    graph_title = f"Box Plot of {x_column}"
                    plt.ylabel(x_column)

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            graph = base64.b64encode(buffer.read()).decode("utf-8")
            plt.close()

        elif visualization == "scatter" and x_column and y_column:

            if x_column in df.columns and y_column in df.columns:

                plt.figure(figsize=(7, 4))
                plt.scatter(df[x_column], df[y_column])
                graph_title = f"{y_column} vs {x_column}"
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format="png")
                buffer.seek(0)
                graph = base64.b64encode(buffer.read()).decode("utf-8")
                plt.close()

        elif visualization == "heatmap":

            if len(numerical_columns) > 1:

                plt.figure(figsize=(8, 6))
                corr = df[numerical_columns].corr()
                sns.heatmap(corr, annot=True, cmap="Blues")
                graph_title = "Correlation Heatmap"
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format="png")
                buffer.seek(0)
                graph = base64.b64encode(buffer.read()).decode("utf-8")
                plt.close()


    # =====================================================
    # BRANCH-WISE STUDENTS
    # =====================================================

    bar_graph = None

    if "branch" in df.columns:

        plt.figure(
            figsize=(6, 4)
        )


        df["branch"].value_counts().plot(
            kind="bar"
        )


        plt.title(
            "Branch Wise Students"
        )

        plt.xlabel(
            "Branch"
        )

        plt.ylabel(
            "Number of Students"
        )


        plt.tight_layout()


        bar_buffer = io.BytesIO()


        plt.savefig(

            bar_buffer,

            format="png"
        )


        bar_buffer.seek(
            0
        )


        bar_graph = base64.b64encode(

            bar_buffer.read()

        ).decode(
            "utf-8"
        )


        plt.close()


    # =====================================================
    # AGE DISTRIBUTION
    # =====================================================

    age_hist_graph = None

    if "age" in df.columns:

        plt.figure(
            figsize=(6, 4)
        )


        plt.hist(

            df["age"].dropna(),

            bins=10,

            edgecolor="black"
        )


        plt.title(
            "Age Distribution"
        )

        plt.xlabel(
            "Age"
        )

        plt.ylabel(
            "Frequency"
        )


        plt.tight_layout()


        age_buffer = io.BytesIO()


        plt.savefig(

            age_buffer,

            format="png"
        )


        age_buffer.seek(
            0
        )


        age_hist_graph = base64.b64encode(

            age_buffer.read()

        ).decode(
            "utf-8"
        )


        plt.close()


    # =====================================================
    # CGPA DISTRIBUTION
    # =====================================================

    cgpa_hist_graph = None

    if "cgpa" in df.columns:

        plt.figure(
            figsize=(6, 4)
        )


        plt.hist(

            df["cgpa"].dropna(),

            bins=10,

            edgecolor="black"
        )


        plt.title(
            "CGPA Distribution"
        )

        plt.xlabel(
            "CGPA"
        )

        plt.ylabel(
            "Frequency"
        )


        plt.tight_layout()


        cgpa_buffer = io.BytesIO()


        plt.savefig(

            cgpa_buffer,

            format="png"
        )


        cgpa_buffer.seek(
            0
        )


        cgpa_hist_graph = base64.b64encode(

            cgpa_buffer.read()

        ).decode(
            "utf-8"
        )


        plt.close()


    # =====================================================
    # AGE BOX PLOT
    # =====================================================

    age_box_graph = None

    if "age" in df.columns:

        plt.figure(
            figsize=(6, 4)
        )


        plt.boxplot(

            df["age"].dropna()
        )


        plt.title(
            "Box Plot of Age"
        )

        plt.ylabel(
            "Age"
        )


        plt.tight_layout()


        age_box_buffer = io.BytesIO()


        plt.savefig(

            age_box_buffer,

            format="png"
        )


        age_box_buffer.seek(
            0
        )


        age_box_graph = base64.b64encode(

            age_box_buffer.read()

        ).decode(
            "utf-8"
        )


        plt.close()


    # =====================================================
    # CORRELATION HEATMAP
    # =====================================================

    heatmap_graph = None

    if len(numerical_columns) > 1:

        plt.figure(
            figsize=(8, 6)
        )


        correlation = df[
            numerical_columns
        ].corr()


        sns.heatmap(

            correlation,

            annot=True,

            cmap="Blues"
        )


        plt.title(
            "Correlation Heatmap"
        )


        plt.tight_layout()


        heatmap_buffer = io.BytesIO()


        plt.savefig(

            heatmap_buffer,

            format="png"
        )


        heatmap_buffer.seek(
            0
        )


        heatmap_graph = base64.b64encode(

            heatmap_buffer.read()

        ).decode(
            "utf-8"
        )


        plt.close()


    return render_template(

        "visualization.html",

        filename=filename,
        columns=columns,
        numerical_columns=numerical_columns,
        graph=graph,
        graph_title=graph_title,
        bar_graph=bar_graph,
        age_hist_graph=age_hist_graph,
        cgpa_hist_graph=cgpa_hist_graph,
        age_box_graph=age_box_graph,
        heatmap_graph=heatmap_graph
    )


@app.route(
    "/linear_regression",
    methods=["GET", "POST"]
)
def linear_regression():

    filename = session.get(
        "dataset"
    )

    if filename is None:

        return redirect(
            "/upload"
        )

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    df = pd.read_csv(filepath)

    page_data = prepare_linear_regression_page(
        df,
        request.form,
        request.method == "POST"
    )

    return render_template(
        "linear_regression.html",
        filename=filename,
        **page_data
    )


# =========================================================
# PLACEMENT PREDICTION
# =========================================================

@app.route(
    "/prediction",
    methods=["GET", "POST"]
)

def prediction():

    filename = session.get(
        "dataset"
    )


    if filename is None:

        return redirect(
            "/upload"
        )


    filepath = os.path.join(

        UPLOAD_FOLDER,

        filename
    )


    df = pd.read_csv(
        filepath
    )


    page_data = prepare_prediction_page(
        df,
        request.method == "POST"
    )

    return render_template(
        "prediction.html",
        filename=filename,
        **page_data
    )


# =========================================================
# LOGISTIC REGRESSION FOR PLACEMENT CLASSIFICATION
# =========================================================

@app.route(
    "/logistic_regression",
    methods=["GET", "POST"]
)
def logistic_regression():

    filename = session.get("dataset")

    if filename is None:

        return redirect("/upload")

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    df = pd.read_csv(filepath)

    page_data = prepare_logistic_regression_page(
        df,
        request.form,
        request.method == "POST"
    )

    return render_template(
        "logistic_regression.html",
        filename=filename,
        **page_data
    )


# =========================================================
# DECISION TREE FOR PLACEMENT CLASSIFICATION
# =========================================================

@app.route(
    "/decision_tree",
    methods=["GET", "POST"]
)
def decision_tree():

    filename = session.get("dataset")

    if filename is None:

        return redirect("/upload")

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    df = pd.read_csv(filepath)

    page_data = prepare_decision_tree_page(
        df,
        request.form,
        request.method == "POST"
    )

    return render_template(
        "decision_tree.html",
        filename=filename,
        **page_data
    )


# =========================================================
# RANDOM FOREST FOR PLACEMENT CLASSIFICATION
# =========================================================

@app.route(
    "/random_forest",
    methods=["GET", "POST"]
)
def random_forest():

    filename = session.get("dataset")

    if filename is None:

        return redirect("/upload")

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    df = pd.read_csv(filepath)

    page_data = prepare_random_forest_page(
        df,
        request.form,
        request.method == "POST"
    )

    return render_template(
        "random_forest.html",
        filename=filename,
        **page_data
    )


# =========================================================
# BOOSTING MODELS FOR PLACEMENT CLASSIFICATION
# =========================================================

@app.route(
    "/boosting",
    methods=["GET", "POST"]
)
def boosting():

    filename = session.get("dataset")

    if filename is None:

        return redirect("/upload")

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    df = pd.read_csv(filepath)

    page_data = prepare_boosting_page(
        df,
        request.form,
        request.method == "POST"
    )

    return render_template(
        "boosting.html",
        filename=filename,
        **page_data
    )


# =========================================================
# K-FOLD CROSS-VALIDATION
# =========================================================

@app.route(
    "/k_fold",
    methods=["GET", "POST"]
)
def k_fold():

    filename = session.get("dataset")

    if filename is None:
        return redirect("/upload")

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    df = pd.read_csv(filepath)
    page_data = prepare_k_fold_page(
        df,
        request.form,
        request.method == "POST"
    )

    return render_template(
        "k_fold.html",
        filename=filename,
        **page_data
    )


# =========================================================
# K-MEANS CLUSTERING
# =========================================================

@app.route(
    "/k_means",
    methods=["GET", "POST"]
)
def k_means():

    filename = session.get("dataset")

    if filename is None:
        return redirect("/upload")

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    df = pd.read_csv(filepath)
    page_data = train_kmeans(
        df,
        request.form,
        request.method == "POST"
    )

    if request.method == "POST" and request.form.get("action") == "predict":
        page_data = predict_kmeans(df, request.form, page_data)

    return render_template(
        "k_means.html",
        filename=filename,
        **page_data
    )


# =========================================================
# PCA AND K-MEANS CLUSTERING
# =========================================================

@app.route(
    "/pca",
    methods=["GET", "POST"]
)
def pca():

    filename = session.get("dataset")

    if filename is None:
        return redirect("/upload")

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    df = pd.read_csv(filepath)
    page_data = run_pca_kmeans(
        df,
        request.form,
        request.method == "POST"
    )

    return render_template(
        "pca.html",
        filename=filename,
        **page_data
    )


# =========================================================
# DIMENSIONALITY REDUCTION COMPARISON
# =========================================================

@app.route(
    "/dimensionality-reduction",
    methods=["GET", "POST"]
)
def dimensionality_reduction_comparison():
    filename = session.get("dataset")

    if filename is None:
        return redirect("/upload")

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    df = pd.read_csv(filepath)
    page_data = run_dimensionality_comparison(
        df,
        request.form,
        request.method == "POST"
    )

    return render_template(
        "dimensionality_comparison.html",
        filename=filename,
        **page_data
    )


# =========================================================
# ROC CURVE CLASSIFIER COMPARISON
# =========================================================

@app.route(
    "/roc-comparison",
    methods=["GET", "POST"]
)
def roc_comparison():
    filename = session.get("dataset")

    if filename is None:
        return redirect("/upload")

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    df = pd.read_csv(filepath)
    page_data = compare_roc_curves(
        df,
        request.form,
        request.method == "POST"
    )

    return render_template(
        "roc_comparison.html",
        filename=filename,
        **page_data
    )


@app.route(
    "/adaboost",
    methods=["GET", "POST"]
)
def adaboost():

    return boosting()


# =========================================================
# RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )