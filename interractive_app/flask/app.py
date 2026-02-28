from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
app.secret_key = "secret-key"
UPLOAD_FOLDER = "uploads"
PLOT_FOLDER = "static/plots"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

sns.set_theme(style="whitegrid")

def get_df():
    path = session.get("csv_path")
    if path is None:
        return None
    return pd.read_csv(path)

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        sep = request.form.get("sep", ",")
        encoding = request.form.get("encoding", "utf-8")

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        df = pd.read_csv(path, sep=sep, encoding=encoding)
        session["csv_path"] = path
        session["shape"] = df.shape

        return redirect(url_for("analyse"))

    return render_template("upload.html")

@app.route("/analyse")
def analyse():
    df = get_df()
    if df is None:
        return redirect(url_for("upload"))

    return render_template(
        "analyse.html",
        head=df.head().to_html(classes="table"),
        shape=df.shape,
        dtypes=df.dtypes.to_frame("Type").to_html(classes="table"),
        missing=df.isna().sum().to_frame("NA").to_html(classes="table"),
        describe=df.describe().to_html(classes="table")
    )

@app.route("/viz", methods=["GET", "POST"])
def viz():
    df = get_df()
    if df is None:
        return redirect(url_for("upload"))

    columns = df.columns.tolist()
    plot_path = None

    if request.method == "POST":
        x = request.form["x"]
        graph = request.form["graph"]

        plt.figure(figsize=(8,4))
        if graph == "hist":
            sns.histplot(df[x], bins=30)
        elif graph == "count":
            sns.countplot(x=df[x])

        plot_path = f"{PLOT_FOLDER}/plot.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return render_template("viz.html", columns=columns, plot=plot_path)

if __name__ == "__main__":
    app.run(debug=True)
