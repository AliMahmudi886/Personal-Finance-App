import os
import sqlite3
from contextlib import closing


# Create a new SQLite database connection
def create_connection():
    conn = sqlite3.connect(os.environ.get("FINANCE_DB_PATH", "personal_finance.db"))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Initialize the database
def init_db():
    with closing(create_connection()) as conn, conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                profile_image TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                date TEXT NOT NULL,
                description TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                goal TEXT NOT NULL,
                target_amount REAL NOT NULL,
                current_amount REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS income (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                amount REAL NOT NULL,
                date TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recurring_expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                frequency TEXT NOT NULL, -- e.g., daily, weekly, monthly
                next_due_date TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
    ensure_column("users", "profile_image")
    ensure_column("expenses", "description")


def ensure_column(table, column):
    # Only internal schema identifiers are passed here.
    with closing(create_connection()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")


def get_user_by_username(username):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cur.fetchone()
    conn.close()
    return user


def create_user(username, password):
    try:
        with closing(create_connection()) as conn, conn:
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password),
            )
    except sqlite3.IntegrityError:
        return False
    return True


def create_income(user_id, source, amount, date):
    with closing(create_connection()) as conn, conn:
        conn.execute(
            "INSERT INTO income (user_id, source, amount, date) VALUES (?, ?, ?, ?)",
            (user_id, source, amount, date),
        )


def get_income_by_user_id(user_id):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM income WHERE user_id = ?", (user_id,))
    income = cur.fetchall()
    conn.close()
    return income


def get_expenses_fortbl_by_user_id(user_id):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM expenses WHERE user_id = ?", (user_id,))
    expenses = cur.fetchall()
    conn.close()
    return expenses


def create_expense(user_id, category, amount, date, description):
    with closing(create_connection()) as conn, conn:
        conn.execute(
            "INSERT INTO expenses (user_id, category, amount, date, description) VALUES (?, ?, ?, ?, ?)",
            (user_id, category, amount, date, description),
        )


def get_goals_by_user_id(user_id):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM goals WHERE user_id = ?", (user_id,))
    goals = cur.fetchall()
    conn.close()
    return goals


def create_goal(user_id, goal, target_amount, current_amount):
    with closing(create_connection()) as conn, conn:
        conn.execute(
            """
            INSERT INTO goals (user_id, goal, target_amount, current_amount)
            VALUES (?, ?, ?, ?)
        """,
            (user_id, goal, target_amount, current_amount),
        )


def get_user_by_id(user_id):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, profile_image FROM users WHERE id = ?", (user_id,)
    )
    user = cursor.fetchone()
    conn.close()

    if user:
        return {"id": user[0], "username": user[1], "profile_image": user[2]}
    return None


def update_user_profile(user_id, username, password=None, profile_image=None):
    with closing(create_connection()) as conn, conn:
        if password and profile_image:
            conn.execute(
                """
                UPDATE users
                SET username = ?, password = ?, profile_image = ?
                WHERE id = ?
            """,
                (username, password, profile_image, user_id),
            )
        elif password:
            conn.execute(
                """
                UPDATE users
                SET username = ?, password = ?
                WHERE id = ?
            """,
                (username, password, user_id),
            )
        elif profile_image:
            conn.execute(
                """
                UPDATE users
                SET username = ?, profile_image = ?
                WHERE id = ?
            """,
                (username, profile_image, user_id),
            )
        else:
            conn.execute(
                """
                UPDATE users
                SET username = ?
                WHERE id = ?
            """,
                (username, user_id),
            )


def get_expenses_by_user_id(user_id):
    conn = create_connection()
    cursor = conn.execute(
        """
        SELECT date, amount FROM expenses WHERE user_id = ? ORDER BY date
    """,
        (user_id,),
    )
    expenses = [{"date": row[0], "amount": row[1]} for row in cursor.fetchall()]
    conn.close()
    return expenses
