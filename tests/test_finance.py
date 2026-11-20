import io
import os
import re
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from PIL import Image


class FinanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.environment.cleanup)
        folder = Path(cls.environment.name)
        cls.variables = patch.dict(
            os.environ,
            FINANCE_SECRET_KEY="synthetic-session-key-for-tests-only",
            FINANCE_DB_PATH=str(folder / "startup.sqlite"),
            FINANCE_HTTPS="0",
        )
        cls.variables.start()
        cls.addClassCleanup(cls.variables.stop)
        from app import app

        cls.app = app

    def setUp(self):
        from models import init_db

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.db_path = Path(folder.name) / "ledger.sqlite"
        variables = patch.dict(os.environ, FINANCE_DB_PATH=str(self.db_path))
        variables.start()
        self.addCleanup(variables.stop)
        settings = patch.dict(
            self.app.config, TESTING=True, UPLOAD_FOLDER=Path(folder.name) / "images"
        )
        settings.start()
        self.addCleanup(settings.stop)
        init_db()
        self.client = self.app.test_client()

    def submit(self, client, path, data, form_path=None):
        page = client.get(form_path or path)
        token = re.search(rb'name="csrf_token" value="([^"]+)"', page.data)
        self.assertIsNotNone(token)
        return client.post(path, data={**data, "csrf_token": token[1].decode()})

    def sign_in(self, client, username):
        credentials = {"username": username, "password": "test-password-example"}
        self.assertEqual(self.submit(client, "/register", credentials).status_code, 302)
        self.assertEqual(self.submit(client, "/login", credentials).status_code, 302)

    def test_expense_write_and_account_isolation(self):
        other = self.app.test_client()
        self.sign_in(self.client, "first-account")
        self.sign_in(other, "second-account")
        entry = {
            "category": "Travel",
            "amount": "25.42",
            "date": "2026-01-01",
            "description": "Synthetic train fare",
        }
        self.assertEqual(
            self.submit(self.client, "/add_expense", entry).status_code, 302
        )
        self.assertIn(b"Synthetic train fare", self.client.get("/view_expenses").data)
        self.assertNotIn(b"Synthetic train fare", other.get("/view_expenses").data)
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            rows = connection.execute(
                "SELECT amount, date, description FROM expenses"
            ).fetchall()
        self.assertEqual(rows, [(25.42, "2026-01-01", "Synthetic train fare")])

    def test_invalid_financial_inputs_leave_database_unchanged(self):
        self.sign_in(self.client, "test-account")
        for amount, entered_date in [
            ("NaN", "2026-01-01"),
            ("1e999", "2026-01-01"),
            ("1e-999", "2026-01-01"),
            ("1", "2026-02-30"),
        ]:
            entry = {
                "category": "Travel",
                "source": "Test",
                "amount": amount,
                "date": entered_date,
            }
            for route in ("/add_expense", "/add_income"):
                self.assertEqual(
                    self.submit(self.client, route, entry).status_code, 400
                )
        self.assertEqual(
            self.submit(
                self.client,
                "/add_goal",
                {"goal": "Fixture", "target_amount": "Infinity", "current_amount": "0"},
            ).status_code,
            400,
        )
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            for table in ("expenses", "income", "goals"):
                self.assertEqual(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0
                )

    def test_active_database_rejects_orphan_expenses(self):
        from models import create_expense

        with self.assertRaises(sqlite3.IntegrityError):
            create_expense(99, "Fixture", "3.50", "2026-01-01", "")

    def test_goal_and_income_forms_and_views(self):
        self.sign_in(self.client, "test-account")
        self.assertEqual(
            self.submit(
                self.client,
                "/add_goal",
                {
                    "goal": "Synthetic goal",
                    "target_amount": "200",
                    "current_amount": "10",
                },
            ).status_code,
            302,
        )
        self.assertIn(b"Synthetic goal", self.client.get("/view_goals").data)
        self.assertEqual(
            self.submit(
                self.client,
                "/add_income",
                {
                    "source": "Synthetic salary",
                    "amount": "100.50",
                    "date": "2026-01-31",
                },
            ).status_code,
            302,
        )
        self.assertIn(b"Synthetic salary", self.client.get("/view_income").data)

    def test_session_and_form_protection(self):
        self.assertEqual(self.client.get("/view_expenses").status_code, 302)
        self.assertEqual(
            self.client.post("/register", data={"username": "fixture"}).status_code, 400
        )
        self.sign_in(self.client, "first-account")
        self.assertEqual(self.client.get("/logout").status_code, 405)
        self.assertEqual(self.client.post("/logout").status_code, 400)
        self.assertEqual(self.submit(self.client, "/logout", {}, "/").status_code, 302)
        self.assertEqual(self.client.get("/view_expenses").status_code, 302)
        with self.client.session_transaction() as session:
            session["user_id"] = 999
        self.assertEqual(self.client.get("/profile").status_code, 302)
        self.assertEqual(self.client.get("/login/google").status_code, 503)

    def test_private_profile_image(self):
        other = self.app.test_client()
        self.sign_in(self.client, "first-account")
        self.sign_in(other, "second-account")
        images = self.app.config["UPLOAD_FOLDER"]
        self.assertEqual(self.client.get("/profile/image").status_code, 404)
        bad = {
            "username": "first-account",
            "profile_image": (io.BytesIO(b"invalid image"), "avatar.png"),
        }
        self.assertEqual(self.submit(self.client, "/profile", bad).status_code, 400)
        self.assertFalse(images.exists())
        stream = io.BytesIO()
        Image.new("RGB", (8, 8)).save(stream, format="PNG")
        data = stream.getvalue()
        conflict = {
            "username": "second-account",
            "profile_image": (io.BytesIO(data), "avatar.png"),
        }
        self.assertEqual(
            self.submit(self.client, "/profile", conflict).status_code, 409
        )
        self.assertEqual(list(images.iterdir()), [])
        image = {
            "username": "first-account",
            "profile_image": (io.BytesIO(data), "avatar.png"),
        }
        self.assertEqual(self.submit(self.client, "/profile", image).status_code, 302)
        with self.client.get("/profile/image") as response:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "image/png")
            self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(other.get("/profile/image").status_code, 404)
        image_path = next(images.iterdir())
        self.assertEqual(
            other.get(f"/static/profile_images/{image_path.name}").status_code, 404
        )

    def test_existing_schema_can_be_opened_twice_without_losing_rows(self):
        from models import init_db

        legacy = self.db_path.with_name("legacy.sqlite")
        with closing(sqlite3.connect(legacy)) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT);
                CREATE TABLE expenses (id INTEGER PRIMARY KEY, user_id INTEGER, category TEXT, amount REAL, date TEXT);
                INSERT INTO users VALUES (1, 'fixture', 'invalid-fixture-hash');
                INSERT INTO expenses VALUES (1, 1, 'Travel', 2.25, '2026-01-01');
                """
            )
        with patch.dict(os.environ, FINANCE_DB_PATH=str(legacy)):
            init_db()
            init_db()
        with closing(sqlite3.connect(legacy)) as connection, connection:
            self.assertEqual(
                connection.execute(
                    "SELECT username, profile_image FROM users"
                ).fetchone(),
                ("fixture", None),
            )
            self.assertEqual(
                connection.execute(
                    "SELECT amount, description FROM expenses"
                ).fetchone(),
                (2.25, None),
            )

    def test_analytics_with_empty_and_small_history(self):
        from ml_model import monthly_expenses

        self.sign_in(self.client, "first-account")
        for route in (
            "/",
            "/predict_expenses",
            "/view_anomalies",
            "/detect_anomalies",
            "/detect_anomalies_autoencoder",
            "/expense_clusters",
            "/recommend_savings",
        ):
            self.assertEqual(self.client.get(route).status_code, 200, route)
        self.assertEqual(
            self.client.get("/expenses/clusters?n_clusters=bad").status_code, 400
        )
        self.assertEqual(
            self.client.get("/expenses/clusters?n_clusters=1000").status_code, 400
        )
        with self.client.session_transaction() as session:
            user_id = session["user_id"]
        for amount in ("0.10", "1.25"):
            self.submit(
                self.client,
                "/add_expense",
                {
                    "amount": amount,
                    "category": "Travel",
                    "date": "2026-01-01",
                    "description": "the",
                },
            )
        self.assertEqual(monthly_expenses(user_id)["amount"].tolist(), [1.35])
        self.assertEqual(self.client.get("/expense_clusters").status_code, 200)

    def test_overview_uses_period_and_account_and_applies_refunds(self):
        from models import create_expense, create_income, get_user_by_username

        self.sign_in(self.client, "overview-owner")
        other = self.app.test_client()
        self.sign_in(other, "overview-other")
        uid = get_user_by_username("overview-owner")[0]
        outsider = get_user_by_username("overview-other")[0]
        create_income(uid, "Salary fixture", "10.30", "2026-01-03")
        create_expense(uid, "Fuel fixture", "3.51", "2026-01-04", "")
        create_expense(uid, "Fuel refund", "-0.21", "2026-01-04", "")
        create_expense(uid, "Outside date", "999", "2026-02-01", "")
        create_expense(outsider, "Other account", "800", "2026-01-04", "")
        period = "/?period=custom&start=2026-01-01&end=2026-01-31"
        response = self.client.get(period)
        self.assertEqual(response.status_code, 200)
        for value in (b"$10.30", b"$3.30", b"$7.00"):
            self.assertIn(value, response.data)
        self.assertNotIn(b"Outside date", response.data)
        self.assertNotIn(b"Other account", response.data)
        self.assertEqual(
            self.client.get(
                "/?period=custom&start=2026-02-01&end=2026-01-01"
            ).status_code,
            400,
        )

    def test_expense_search_and_page_boundary(self):
        from models import create_expense, get_user_by_username

        self.sign_in(self.client, "page-owner")
        other = self.app.test_client()
        self.sign_in(other, "page-other")
        uid = get_user_by_username("page-owner")[0]
        outsider = get_user_by_username("page-other")[0]
        for number in range(41):
            create_expense(
                uid, "Rail", "2.50", "2026-01-01", f"Search fixture {number:02}"
            )
        create_expense(outsider, "Rail", "2.50", "2026-01-01", "Search fixture outside")
        first = self.client.get("/view_expenses?q=search+fixture").data
        second = self.client.get("/view_expenses?q=search+fixture&page=2").data
        self.assertIn(b"Page 1 of 2", first)
        self.assertIn(b"Search fixture 40", first)
        self.assertNotIn(b"Search fixture 00", first)
        self.assertIn(b"Search fixture 00", second)
        self.assertNotIn(b"Search fixture outside", first + second)
        self.assertIn(
            b"No expenses match your search",
            self.client.get("/view_expenses?q=%25").data,
        )

    def test_spending_reference_shows_latest_expense_including_zero(self):
        from models import create_expense, get_user_by_username

        self.sign_in(self.client, "first-account")
        other = self.app.test_client()
        self.sign_in(other, "second-account")
        first = get_user_by_username("first-account")[0]
        second = get_user_by_username("second-account")[0]
        self.assertIn(b"Record an expense", self.client.get("/recommend_savings").data)
        create_expense(first, "Fixture", "4.20", "2026-01-02", "")
        create_expense(first, "Fixture", "0", "2026-01-02", "")
        create_expense(first, "Fixture", "30", "2026-01-01", "")
        create_expense(second, "Fixture", "78.15", "2026-01-03", "")
        page = self.client.get("/recommend_savings")
        self.assertIn(b"$0.00", page.data)
        self.assertIn(b"2026-01-02", page.data)
        self.assertNotIn(b"78.15", page.data)

    def test_analytics_reports_invalid_legacy_rows_without_changing_them(self):
        self.sign_in(self.client, "test-account")
        with self.client.session_transaction() as session:
            user_id = session["user_id"]
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute(
                "INSERT INTO expenses (user_id, category, amount, date) VALUES (?, ?, ?, ?)",
                (user_id, "Legacy fixture", 5, "invalid-date"),
            )
        response = self.client.get("/predict_expenses")
        self.assertEqual(response.status_code, 422)
        self.assertIn(b"Expense analytics unavailable", response.data)
        self.assertIn(b"invalid-date", self.client.get("/view_expenses").data)

    @unittest.skipUnless(
        os.environ.get("FINANCE_TEST_ML") == "1", "requires TensorFlow runtime"
    )
    def test_tensorflow_analytics_run_with_synthetic_history(self):
        self.sign_in(self.client, "test-account")
        for month, amount in enumerate(("2", "4", "3"), start=1):
            self.submit(
                self.client,
                "/add_expense",
                {
                    "amount": amount,
                    "category": "Travel",
                    "date": f"2026-{month:02}-01",
                    "description": "train ticket",
                },
            )
        self.assertIn(b"Next Month", self.client.get("/predict_expenses").data)
        self.assertEqual(
            self.client.get("/detect_anomalies_autoencoder").status_code, 200
        )


if __name__ == "__main__":
    unittest.main()
