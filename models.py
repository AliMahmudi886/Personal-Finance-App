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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS expenses_user_date_idx ON expenses (user_id, date)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS income_user_idx ON income (user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS goals_user_idx ON goals (user_id)")
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
    cur.execute(
        "SELECT * FROM income WHERE user_id = ? ORDER BY date DESC, id DESC", (user_id,)
    )
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


def get_latest_expense(user_id):
    with closing(create_connection()) as conn:
        return conn.execute(
            "SELECT date, amount FROM expenses WHERE user_id = ? ORDER BY date DESC, id DESC LIMIT 1",
            (user_id,),
        ).fetchone()


def get_expense_page(user_id, query, page):
    size = 40
    where = "WHERE user_id = ?"
    values = [user_id]
    if query:
        where += " AND (instr(lower(category), lower(?)) > 0 OR instr(lower(description), lower(?)) > 0)"
        values.extend((query, query))
    with closing(create_connection()) as db:
        db.execute("BEGIN")
        total = db.execute(f"SELECT count(*) FROM expenses {where}", values).fetchone()[
            0
        ]
        pages = max(1, (total + size - 1) // size)
        page = min(page, pages)
        rows = db.execute(
            f"""SELECT id, user_id, category, amount, date, description FROM expenses {where}
                ORDER BY date DESC, id DESC LIMIT ? OFFSET ?""",
            (*values, size, (page - 1) * size),
        ).fetchall()
    return rows, page, pages, total


def get_goal(user_id, goal_id):
    with closing(create_connection()) as db:
        return db.execute(
            "SELECT goal, target_amount, current_amount FROM goals WHERE user_id = ? AND id = ?",
            (user_id, goal_id),
        ).fetchone()


def update_goal(user_id, goal_id, name, target, current):
    with closing(create_connection()) as db, db:
        result = db.execute(
            "UPDATE goals SET goal = ?, target_amount = ?, current_amount = ? WHERE user_id = ? AND id = ?",
            (name, target, current, user_id, goal_id),
        )
        return result.rowcount == 1


def get_recurring_expenses(user_id):
    with closing(create_connection()) as db:
        return db.execute(
            """SELECT id, category, amount, frequency, next_due_date FROM recurring_expenses
               WHERE user_id = ? ORDER BY next_due_date, id""",
            (user_id,),
        ).fetchall()


def get_recurring_expense(user_id, reminder_id):
    with closing(create_connection()) as db:
        return db.execute(
            """SELECT category, amount, frequency, next_due_date FROM recurring_expenses
               WHERE user_id = ? AND id = ?""",
            (user_id, reminder_id),
        ).fetchone()


def save_recurring_expense(user_id, reminder_id, category, amount, frequency, due):
    with closing(create_connection()) as db, db:
        fields = (category, amount, frequency, due, user_id)
        if reminder_id is None:
            db.execute(
                """INSERT INTO recurring_expenses (category, amount, frequency, next_due_date, user_id)
                   VALUES (?, ?, ?, ?, ?)""",
                fields,
            )
            return True
        result = db.execute(
            """UPDATE recurring_expenses SET category = ?, amount = ?, frequency = ?, next_due_date = ?
               WHERE user_id = ? AND id = ?""",
            (*fields, reminder_id),
        )
        return result.rowcount == 1
