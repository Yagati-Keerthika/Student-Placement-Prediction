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
from main import train_model


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
    scale_data
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


    table = None

    message = None

    missing_summary = get_missing_values(df).to_html(
        classes="table table-bordered"
    )

    missing_columns = get_missing_columns(df)

    code_example = None


    # =====================================================
    # APPLY MISSING VALUE HANDLING
    # =====================================================

    if request.method == "POST":

        method = request.form.get(
            "method",
            "mean"
        )

        selected_columns = request.form.getlist(
            "columns"
        )

        if not selected_columns:

            selected_columns = missing_columns

        if method == "all":

            df = fill_missing_values(
                df,
                "all",
                selected_columns
            )

            message = (
                "Missing values filled with the default "
                "mean for numeric columns and mode for "
                "categorical columns."
            )

        else:

            df = fill_missing_values(
                df,
                method,
                selected_columns
            )

            message = (
                f"Selected columns were filled using "
                f"{method.upper()}."
            )

        code_example = (
            "df = df.copy()\n"
            f"columns = {selected_columns}\n"
            f"method = '{method}'\n"
            "for column in columns:\n"
            "    if column in df.columns and df[column].isnull().sum() > 0:\n"
            "        if pd.api.types.is_numeric_dtype(df[column]):\n"
            "            if method == 'mean':\n"
            "                df[column] = df[column].fillna(df[column].mean())\n"
            "            elif method == 'median':\n"
            "                df[column] = df[column].fillna(df[column].median())\n"
            "        else:\n"
            "            df[column] = df[column].fillna(df[column].mode().iloc[0])\n"
        )

        # =================================================
        # SAVE PROCESSED DATASET
        # =================================================

        processed_filename = (
            "missing_" + filename
        )


        processed_path = os.path.join(

            PROCESSED_FOLDER,

            processed_filename
        )


        df.to_csv(

            processed_path,

            index=False
        )


        # =================================================
        # PREVIEW
        # =================================================

        table = df.head(
            10
        ).to_html(

            classes="table table-bordered",

            index=False
        )

    else:

        code_example = (
            "# Example:\n"
            "# df = df.copy()\n"
            "# columns = ['CGPA', 'Gender']\n"
            "# for column in columns:\n"
            "#     if df[column].isnull().sum() > 0:\n"
            "#         df[column] = df[column].fillna(df[column].mean())\n"
        )


    return render_template(

        "missing_value.html",

        filename=filename,

        missing_summary=missing_summary,

        missing_columns=missing_columns,

        table=table,

        message=message,

        code_example=code_example
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


    table = None

    message = None

    code_example = None

    categorical_columns = get_categorical_columns(
        df
    )


    # =====================================================
    # APPLY ENCODING
    # =====================================================

    if request.method == "POST":

        encoding_method = request.form.get(
            "method",
            "label"
        )

        selected_columns = request.form.getlist(
            "columns"
        )

        if not selected_columns:

            selected_columns = categorical_columns

        df = encode_data(
            df,
            encoding_method=encoding_method,
            columns=selected_columns
        )

        message = (
            f"Categorical data encoded using "
            f"{encoding_method.upper()} encoding."
        )

        code_example = (
            "# Example:\n"
            f"columns = {selected_columns}\n"
            f"method = '{encoding_method}'\n"
            "if method == 'onehot':\n"
            "    df = pd.get_dummies(df, columns=columns)\n"
            "else:\n"
            "    for column in columns:\n"
            "        df[column] = df[column].astype('category').cat.codes\n"
        )

        # =================================================
        # SAVE PROCESSED DATASET
        # =================================================

        processed_filename = (
            "encoded_" + filename
        )


        processed_path = os.path.join(

            PROCESSED_FOLDER,

            processed_filename
        )


        df.to_csv(

            processed_path,

            index=False
        )


        # =================================================
        # PREVIEW
        # =================================================

        table = df.head(
            10
        ).to_html(

            classes="table table-bordered",

            index=False
        )


    return render_template(

        "encoding.html",

        filename=filename,

        categorical_columns=categorical_columns,

        table=table,

        message=message,

        code_example=code_example
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


    table = None

    method_name = None


    # =====================================================
    # NUMERICAL COLUMNS
    # =====================================================

    numerical_columns = get_numerical_columns(
        df
    )


    # =====================================================
    # APPLY SCALING
    # =====================================================

    if request.method == "POST":

        scaling_method = request.form.get(
            "scaling_method"
        )

        selected_columns = request.form.getlist(
            "columns"
        )

        if not selected_columns:

            selected_columns = numerical_columns


        if scaling_method == "standard":

            method_name = (
                "Standard Scaler"
            )


        elif scaling_method == "minmax":

            method_name = (
                "Min-Max Scaler"
            )


        elif scaling_method == "normalize":

            method_name = (
                "Normalization"
            )


        else:

            return (
                "Please select a scaling method."
            )


        # =================================================
        # APPLY SCALING
        # =================================================

        df = scale_data(

            df,

            scaling_method,

            selected_columns
        )


        # =================================================
        # SAVE PROCESSED DATASET
        # =================================================

        processed_filename = (
            "scaled_" + filename
        )


        processed_path = os.path.join(

            PROCESSED_FOLDER,

            processed_filename
        )


        df.to_csv(

            processed_path,

            index=False
        )


        # =================================================
        # SCALED DATASET PREVIEW
        # =================================================

        table = df.head(
            10
        ).to_html(

            classes="table table-bordered",

            index=False
        )


    return render_template(

        "scaling.html",

        filename=filename,

        numerical_columns=numerical_columns,

        table=table,

        method_name=method_name
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


    prediction_result = None

    accuracy = None


    # =====================================================
    # TARGET COLUMN
    # =====================================================

    target_column = None


    possible_targets = [

        "placement",

        "placed",

        "status",

        "placement_status"

    ]


    for column in possible_targets:

        if column in df.columns:

            target_column = column

            break


    if target_column is None:

        return render_template(

            "prediction.html",

            error="Placement target column not found."
        )


    # =====================================================
    # TRAIN MODEL
    # =====================================================

    if request.method == "POST":

        model, accuracy = train_model(

            df,

            target_column
        )


        prediction_result = (

            "Model trained successfully."
        )


    return render_template(

        "prediction.html",

        filename=filename,

        prediction_result=prediction_result,

        accuracy=accuracy,

        target_column=target_column
    )


# =========================================================
# RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )