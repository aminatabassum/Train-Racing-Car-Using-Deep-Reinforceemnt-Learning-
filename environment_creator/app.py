from flask import Flask, request, render_template, jsonify, session
import numpy as np

# Instantiate flask app
app = Flask(__name__)

# Basic config for flask app
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.secret_key = "my-secret-key"
app.config["SESSION_TYPE"] = "filesystem"


@app.route("/", methods=["GET"])
def index():

    if request.method == "GET":
        return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():

    if request.method == "POST":
        nr = request.form["nr"]
        nc = request.form["nc"]

        session["nr"] = int(nr)
        session["nc"] = int(nc)

        return jsonify(
            {
                "icon": "success",
                "title": "Success",
                "text": f"Generated {nr}x{nc} grid!",
                "url": "/grid",
            }
        )


@app.route("/edit", methods=["POST"])
def edit():

    if request.method == "POST":
        path = request.form["path"]

        with open(path, "r") as f:
            grid = np.loadtxt(f, dtype=int)

        session["grid"] = grid.tolist()
        session["nr"], session["nc"] = grid.shape

        return jsonify(
            {
                "icon": "success",
                "title": "Success",
                "text": f"Loaded grid from {path}!",
                "url": "/grid",
            }
        )


@app.route("/back", methods=["POST"])
def back():

    if request.method == "POST":
        session.pop("grid", None)
        session.pop("nr", None)
        session.pop("nc", None)

        return jsonify(
            {
                "icon": "success",
                "title": "Success",
                "text": "Back to home!",
                "url": "/",
            }
        )


@app.route("/grid", methods=["GET"])
def grid():

    if request.method == "GET":
        nr = session.get("nr")
        nc = session.get("nc")

        session["grid"] = (
            [[1 for _ in range(nc)] for _ in range(nr)]
            if session.get("grid") is None
            else session.get("grid")
        )

        return render_template("grid.html", grid=session["grid"])


@app.route("/update-grid", methods=["GET", "POST"])
def update_grid():

    if request.method == "GET":
        return render_template("grid.html", grid=session["grid"], val=session["val"])
    elif request.method == "POST":
        r = request.form["r"]
        c = request.form["c"]
        val = request.form["val"]

        grid = session.get("grid")

        grid[int(r)][int(c)] = int(val)

        session["grid"] = grid
        session["val"] = val

        return jsonify(
            {
                "icon": "success",
                "title": "Success",
                "text": f"Updated grid at ({r}, {c})!",
                "url": "/update-grid",
            }
        )


@app.route("/reset-grid", methods=["POST"])
def reset():

    if request.method == "POST":
        session["val"] = 1
        session["grid"] = [
            [1 for _ in range(session["nc"])] for _ in range(session["nr"])
        ]

        return jsonify(
            {
                "icon": "success",
                "title": "Success",
                "text": "Reset grid!",
                "url": "/update-grid",
            }
        )


@app.route("/save-grid", methods=["POST"])
def save():

    if request.method == "POST":
        save_path = request.form["save-path"]
        grid = session.get("grid")
        grid_arr = np.array(grid)

        with open(save_path, "w") as f:
            np.savetxt(f, grid_arr, fmt="%d")

        return jsonify(
            {
                "icon": "success",
                "title": "Success",
                "text": f"Saved grid to {save_path}!",
            }
        )


if __name__ == "__main__":
    app.run(debug=True)
