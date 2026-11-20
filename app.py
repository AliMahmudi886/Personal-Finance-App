import os
import sqlite3
from datetime import date
from decimal import Decimal, InvalidOperation
from math import isfinite
from pathlib import Path
from uuid import uuid4

from flask import (
    Flask,
    abort,
    flash,
    g,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from flask_bcrypt import Bcrypt
from flask_wtf.csrf import CSRFProtect
from PIL import Image

from forecast import MIN_HISTORY, forecast_for_user, month_index, month_label
from ml_model import (
    InvalidExpenseHistory,
    cluster_expenses,
    detect_anomalies,
    detect_anomalies_autoencoder,
    train_autoencoder,
)
from models import (
    create_expense,
    create_goal,
    create_income,
    create_user,
    get_expense_page,
    get_goal,
    get_goals_by_user_id,
    get_income_by_user_id,
    get_latest_expense,
    get_recurring_expense,
    get_recurring_expenses,
    get_user_by_id,
    get_user_by_username,
    init_db,
    save_recurring_expense,
    update_goal,
    update_user_profile,
)
from overview import financial_overview

app = Flask(__name__)
secret_key = os.environ.get("FINANCE_SECRET_KEY", "")
if len(secret_key) < 32:
    raise RuntimeError(
        "Set FINANCE_SECRET_KEY to a random value of at least 32 characters."
    )
app.config.update(
    SECRET_KEY=secret_key,
    SESSION_COOKIE_SECURE=os.environ.get("FINANCE_HTTPS", "1") != "0",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,
    UPLOAD_FOLDER=Path(app.instance_path) / "profile_images",
)
bcrypt = Bcrypt(app)
CSRFProtect(app)
init_db()


@app.after_request
def protect_response(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.endpoint != "static":
        response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/login/google")
@app.route("/login/google/authorized")
def google_login():
    abort(
        503,
        description="Google sign-in is unavailable. Sign in with your username and password.",
    )


def transaction_date(value):
    return date.fromisoformat(value).isoformat()


@app.errorhandler(InvalidExpenseHistory)
def invalid_expense_history(error):
    return render_template("analytics_error.html"), 422


def checked_amount(value):
    amount = Decimal(value)
    numeric = float(amount) if amount.is_finite() else float("nan")
    if not isfinite(numeric) or (numeric == 0 and not amount.is_zero()):
        raise ValueError("Enter a finite amount that fits the ledger format.")
    return str(amount)


@app.before_request
def load_user():
    if request.endpoint == "static":
        filename = request.view_args.get("filename", "")
        top_folder = filename.replace("\\", "/").split("/", 1)[0].strip().casefold()
        if top_folder == "profile_images":
            abort(404)
        return
    user_id = session.get("user_id")
    g.user = get_user_by_id(user_id) if type(user_id) is int else None
    if "user_id" in session and g.user is None:
        session.clear()


@app.route("/")
def home():
    if not g.user:
        return render_template("home.html")
    today = date.today()
    period = request.args.get("period", "month")
    if period not in {"month", "quarter", "year", "custom"}:
        abort(400, description="Choose a valid reporting period.")
    start, end = today.replace(day=1), today
    if period == "quarter":
        year, month = divmod(today.year * 12 + today.month - 3, 12)
        start = date(year, month + 1, 1)
    elif period == "year":
        start = today.replace(month=1, day=1)
    elif period == "custom":
        try:
            start = date.fromisoformat(request.args.get("start", ""))
            end = date.fromisoformat(request.args.get("end", ""))
        except ValueError:
            abort(400, description="Enter both dates in the reporting range.")
        if not 0 <= (end - start).days <= 730:
            abort(400, description="Choose a date range up to 731 days.")
    report = None
    report_error = False
    try:
        report = financial_overview(g.user["id"], start.isoformat(), end.isoformat())
    except (InvalidOperation, ValueError):
        report_error = True
    return render_template(
        "home.html",
        report=report,
        report_error=report_error,
        period=period,
        start=start,
        end=end,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = get_user_by_username(username)
        try:
            valid = user and bcrypt.check_password_hash(user[2], password)
        except ValueError:
            valid = False
        if valid:
            session.clear()
            session["user_id"] = user[0]
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials", "danger")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if not username.strip() or not 1 <= len(password.encode("utf-8")) <= 72:
            flash("Enter a username and a password of 1 to 72 UTF-8 bytes.", "danger")
            return render_template("register.html"), 400
        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
        if create_user(username, hashed_password):
            flash("Registration successful!", "success")
            return redirect(url_for("login"))
        else:
            flash("Registration failed. Username may already be taken.", "danger")
    return render_template("register.html")


@app.post("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/add_expense", methods=["GET", "POST"])
def add_expense():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        try:
            category = required_label("category")
            amount = checked_amount(request.form["amount"])
            entered_date = transaction_date(request.form["date"])
        except (InvalidOperation, ValueError):
            flash("Enter a name, a valid amount and a valid date.", "danger")
            return render_template(
                "add_expense.html", today=date.today().isoformat()
            ), 400
        create_expense(
            session["user_id"],
            category,
            amount,
            entered_date,
            request.form.get("description", ""),
        )
        flash("Expense added!", "success")
        return redirect(url_for("view_expenses"))
    return render_template("add_expense.html", today=date.today().isoformat())


@app.route("/view_expenses")
def view_expenses():
    if "user_id" not in session:
        return redirect(url_for("login"))
    query = request.args.get("q", "").strip()[:100]
    page = max(request.args.get("page", 1, type=int), 1)
    expenses, page, pages, total = get_expense_page(g.user["id"], query, page)
    return render_template(
        "view_expenses.html",
        expenses=expenses,
        q=query,
        page=page,
        pages=pages,
        total=total,
    )


def required_label(name):
    value = request.form.get(name, "").strip()
    if not value or len(value) > 100:
        raise ValueError("Enter a name between 1 and 100 characters.")
    return value


@app.route("/add_goal", methods=["GET", "POST"])
@app.route("/goals/<int:goal_id>/edit", methods=["GET", "POST"])
def add_goal(goal_id=None):
    if not g.user:
        return redirect(url_for("login"))
    uid = g.user["id"]
    record = get_goal(uid, goal_id) if goal_id is not None else ("", "", "0")
    if record is None:
        abort(404)
    if request.method == "POST":
        try:
            name = required_label("goal")
            target = checked_amount(request.form.get("target_amount", ""))
            current = checked_amount(request.form.get("current_amount", ""))
            if Decimal(target) <= 0 or Decimal(current) < 0:
                raise ValueError
        except (ValueError, InvalidOperation):
            flash(
                "Enter a goal name, a positive target and a nonnegative saved amount.",
                "danger",
            )
            return render_template(
                "add_goal.html", record=record, editing=goal_id is not None
            ), 400
        if goal_id is None:
            create_goal(uid, name, target, current)
            flash("Goal added!", "success")
        elif update_goal(uid, goal_id, name, target, current):
            flash("Goal updated!", "success")
        else:
            abort(404)
        return redirect(url_for("view_goals"))
    return render_template("add_goal.html", record=record, editing=goal_id is not None)


@app.route("/view_goals")
def view_goals():
    if not g.user:
        return redirect(url_for("login"))
    goals = []
    for goal_id, _, name, target, current in get_goals_by_user_id(g.user["id"]):
        progress = None
        if (
            isinstance(target, (float, int))
            and isinstance(current, (float, int))
            and isfinite(target)
            and isfinite(current)
            and target > 0
            and current >= 0
        ):
            progress = (
                100
                if current >= target
                else int(Decimal(str(current)) / Decimal(str(target)) * 100)
            )
        goals.append((goal_id, name, target, current, progress))
    return render_template("view_goals.html", goals=goals)


@app.get("/recurring")
def recurring_expenses():
    if not g.user:
        return redirect(url_for("login"))
    return render_template(
        "recurring_expenses.html",
        reminders=get_recurring_expenses(g.user["id"]),
        today=date.today().isoformat(),
    )


@app.route("/recurring/add", methods=["GET", "POST"])
@app.route("/recurring/<int:reminder_id>/edit", methods=["GET", "POST"])
def recurring_form(reminder_id=None):
    if not g.user:
        return redirect(url_for("login"))
    uid = g.user["id"]
    frequencies = ("daily", "weekly", "monthly", "yearly")
    editing = reminder_id is not None
    record = (
        get_recurring_expense(uid, reminder_id)
        if editing
        else ("", "", "monthly", date.today().isoformat())
    )
    if record is None:
        abort(404)
    if request.method == "POST":
        try:
            category = required_label("category")
            amount = checked_amount(request.form.get("amount", ""))
            frequency = request.form.get("frequency")
            due = transaction_date(request.form.get("next_due_date", ""))
            if Decimal(amount) <= 0 or frequency not in frequencies:
                raise ValueError
        except (ValueError, InvalidOperation):
            flash(
                "Enter a reminder name, a positive amount, a frequency and a valid date.",
                "danger",
            )
            return render_template(
                "recurring_form.html",
                record=record,
                editing=editing,
                frequencies=frequencies,
            ), 400
        if not save_recurring_expense(
            uid, reminder_id, category, amount, frequency, due
        ):
            abort(404)
        flash("Reminder saved.", "success")
        return redirect(url_for("recurring_expenses"))
    return render_template(
        "recurring_form.html", record=record, editing=editing, frequencies=frequencies
    )


@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        username = request.form["username"]
        password = request.form.get("password", "")
        if not username.strip() or len(password.encode("utf-8")) > 72:
            flash(
                "Enter a username and a password of at most 72 UTF-8 bytes.", "danger"
            )
            return render_template("profile.html", user=g.user), 400
        hashed_password = (
            bcrypt.generate_password_hash(password).decode("utf-8")
            if password
            else None
        )
        upload = request.files.get("profile_image")
        image_path = None

        if upload and upload.filename:
            try:
                image_path = save_profile_image(upload)
            except (OSError, ValueError, Image.DecompressionBombError):
                flash(
                    "Choose a valid JPEG, PNG or WebP image up to 4 megapixels.",
                    "danger",
                )
                return render_template("profile.html", user=g.user), 400

        try:
            update_user_profile(
                g.user["id"],
                username,
                hashed_password,
                image_path.name if image_path else None,
            )
        except Exception as error:
            if image_path:
                image_path.unlink(missing_ok=True)
            if not isinstance(error, sqlite3.IntegrityError):
                raise
            flash("That username is already taken.", "danger")
            return render_template("profile.html", user=g.user), 409

        flash("Profile updated!", "success")
        return redirect(url_for("profile"))

    return render_template("profile.html", user=g.user)


def save_profile_image(upload):
    with Image.open(upload.stream, formats=("JPEG", "PNG", "WEBP")) as image:
        if image.width * image.height > 4_000_000:
            raise ValueError("Image is too large.")
        pixels = Image.new("RGBA", image.size)
        pixels.paste(image.convert("RGBA"))
        folder = Path(app.config["UPLOAD_FOLDER"])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{uuid4().hex}.png"
        try:
            with path.open("xb") as output:
                pixels.save(output, format="PNG")
        except OSError:
            path.unlink(missing_ok=True)
            raise
    return path


@app.get("/profile/image")
def profile_image():
    if not g.user or not g.user["profile_image"]:
        abort(404)
    filename = g.user["profile_image"]
    if Path(filename).name != filename or Path(filename).suffix.lower() not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }:
        abort(404)
    folder = Path(app.config["UPLOAD_FOLDER"])
    if not (folder / filename).is_file():
        folder = Path(app.static_folder) / "profile_images"
    return send_from_directory(folder, filename)


@app.route("/add_income", methods=["GET", "POST"])
def add_income():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        try:
            source = required_label("source")
            amount = checked_amount(request.form["amount"])
            entered_date = transaction_date(request.form["date"])
        except (InvalidOperation, ValueError):
            flash("Enter a name, a valid amount and a valid date.", "danger")
            return render_template(
                "add_income.html", today=date.today().isoformat()
            ), 400
        create_income(session["user_id"], source, amount, entered_date)
        flash("Income added!", "success")
        return redirect(url_for("view_income"))

    return render_template("add_income.html", today=date.today().isoformat())


@app.route("/view_income")
def view_income():
    if "user_id" not in session:
        return redirect(url_for("login"))

    income = get_income_by_user_id(session["user_id"])
    return render_template("view_income.html", income=income)


@app.route("/predict_expenses", methods=["GET", "POST"])
def predict_expenses():
    if not g.user:
        return redirect(url_for("login"))
    today = date.today()
    try:
        forecast, closed_count = forecast_for_user(g.user["id"], today)
    except ValueError as error:
        raise InvalidExpenseHistory from error

    scenario = None
    scenario_error = None
    adjustment_input = ""
    if request.method == "POST":
        adjustment_input = request.form.get("adjustment", "")
        if forecast is None or request.form.get("target") != forecast.target:
            scenario_error = "The forecast period changed or is unavailable. Review the current forecast."
        else:
            try:
                adjustment = Decimal(checked_amount(adjustment_input))
                if adjustment != adjustment.quantize(Decimal("0.01")):
                    raise ValueError
                scenario = forecast.amount + adjustment
            except (InvalidOperation, ValueError):
                scenario_error = (
                    "Enter an adjustment with no more than two decimal places."
                )

    chart = None
    if forecast is not None:
        history = forecast.history[-24:]
        chart = {
            "labels": [label for label, _ in history]
            + [month_label(month_index(today)), forecast.target],
            "actual": [float(value) for _, value in history] + [None, None],
            "estimate": [None] * (len(history) + 1) + [float(forecast.amount)],
        }
    return render_template(
        "predict_expenses.html",
        forecast=forecast,
        chart=chart,
        closed_count=closed_count,
        minimum=MIN_HISTORY,
        scenario=scenario,
        scenario_error=scenario_error,
        adjustment_input=adjustment_input,
    ), 400 if scenario_error else 200


@app.route("/view_anomalies")
def view_anomalies():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]
    anomalies = detect_anomalies(user_id)

    return render_template("view_anomalies.html", anomalies=anomalies)


app.add_url_rule("/detect_anomalies", view_func=view_anomalies)


@app.route("/detect_anomalies_autoencoder")
def detect_anomalies_autoencoder_route():
    if not g.user:
        return redirect(url_for("login"))
    user_id = g.user["id"]
    model, scaler = train_autoencoder(user_id)
    if model is None:
        return render_template(
            "view_anomalies.html", error="Not enough data to detect anomalies."
        )
    anomalies = detect_anomalies_autoencoder(user_id, model, scaler)
    return render_template("view_anomalies.html", anomalies=anomalies)


@app.route("/expense_clusters")
def view_expense_clusters():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]
    try:
        clustered_data = cluster_expenses(user_id)
        if clustered_data is None:
            return render_template("view_clusters.html", clustered_data=[])
        clustered_data = clustered_data.to_dict(
            orient="records"
        )  # Convert DataFrame to list of dictionaries
    except ValueError as e:
        flash(str(e), "danger")
        return redirect(url_for("home"))

    return render_template("view_clusters.html", clustered_data=clustered_data)


@app.route("/recommend_savings")
def recommend_savings():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]
    latest = get_latest_expense(user_id)
    if latest is not None:
        entered_date, amount = latest
        try:
            transaction_date(entered_date)
        except ValueError as error:
            raise InvalidExpenseHistory from error
        if not isinstance(amount, (int, float)) or not isfinite(amount):
            raise InvalidExpenseHistory
    return render_template("recommend_savings.html", latest=latest)


@app.route("/expenses/clusters")
def expense_clusters():
    if "user_id" in session:
        user_id = session["user_id"]
        method = request.args.get("method", "kmeans")
        try:
            n_clusters = int(request.args.get("n_clusters", 3))
            clustered_data = cluster_expenses(
                user_id, method=method, n_clusters=n_clusters
            )
        except ValueError:
            abort(
                400,
                description="Select a supported method and a cluster count from 2 to 20.",
            )
        if clustered_data is not None:
            clusters = {
                label: rows.to_dict(orient="records")
                for label, rows in clustered_data.groupby("cluster")
            }
            return render_template(
                "expense_clusters.html",
                clusters=clusters,
                method=method,
                n_clusters=n_clusters,
            )

        return render_template(
            "expense_clusters.html", clusters={}, method=method, n_clusters=n_clusters
        )

    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run()
