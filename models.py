import sqlite3
import os

# Create a new SQLite database connection
def create_connection():
    conn = None
    try:
        conn = sqlite3.connect('personal_finance.db')
    except sqlite3.Error as e:
        print(e)
    return conn

# Initialize the database
def init_db():
    conn = create_connection()
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                profile_image TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                date TEXT NOT NULL,
                description TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                goal TEXT NOT NULL,
                target_amount REAL NOT NULL,
                current_amount REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS income (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                amount REAL NOT NULL,
                date TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS recurring_expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                frequency TEXT NOT NULL, -- e.g., daily, weekly, monthly
                next_due_date TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.execute('''
                  CREATE TABLE IF NOT EXISTS recurring_expenses (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER NOT NULL,
                      category TEXT NOT NULL,
                      amount REAL NOT NULL,
                      frequency TEXT NOT NULL, -- e.g., daily, weekly, monthly
                      next_due_date TEXT NOT NULL,
                      FOREIGN KEY (user_id) REFERENCES users (id)
                  )
              ''')
    conn.close()
def add_description_column():
    conn = create_connection()
    try:
        with conn:
            conn.execute('ALTER TABLE expenses ADD COLUMN description TEXT')
    except sqlite3.OperationalError as e:
        if 'duplicate column name: description' not in str(e):
            print(e)
    finally:
        conn.close()
def add_profile_image_column():
    conn = create_connection()
    try:
        with conn:
            conn.execute('ALTER TABLE users ADD COLUMN profile_image TEXT')
    except sqlite3.OperationalError as e:
        if 'duplicate column name: profile_image' not in str(e):
            raise
    finally:
        conn.close()

# Call this function to apply the schema change
add_profile_image_column()

def get_user_by_username(username):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cur.fetchone()
    conn.close()
    return user

def create_user(username, password):
    conn = create_connection()
    try:
        with conn:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        return True
    except sqlite3.IntegrityError:
        return False

def create_income(user_id, source, amount, date):
    conn = create_connection()
    with conn:
        conn.execute('INSERT INTO income (user_id, source, amount, date) VALUES (?, ?, ?, ?)', (user_id, source, amount, date))
    conn.close()

def get_income_by_user_id(user_id):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM income WHERE user_id = ?', (user_id,))
    income = cur.fetchall()
    conn.close()
    return income

def get_expenses_fortbl_by_user_id(user_id):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM expenses WHERE user_id = ?', (user_id,))
    expenses = cur.fetchall()
    conn.close()
    return expenses

def create_expense(user_id, category, amount, date, description):
    conn = create_connection()
    with conn:
        conn.execute('INSERT INTO expenses (user_id, category, amount, date, description) VALUES (?, ?, ?, ?, ?)', (user_id, category, amount, date, description))
    conn.close()

def get_goals_by_user_id(user_id):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM goals WHERE user_id = ?', (user_id,))
    goals = cur.fetchall()
    conn.close()
    return goals

def create_goal(user_id, goal, target_amount, current_amount):
    conn = create_connection()
    with conn:
        conn.execute('''
            INSERT INTO goals (user_id, goal, target_amount, current_amount)
            VALUES (?, ?, ?, ?)
        ''', (user_id, goal, target_amount, current_amount))
    conn.close()

def get_user_by_id(user_id):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, profile_image FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    if user:
        return {'id': user[0], 'username': user[1], 'profile_image': user[2]}
    return None

def update_user_profile(user_id, username, password=None, profile_image=None):
    conn = create_connection()
    with conn:
        if password and profile_image:
            conn.execute('''
                UPDATE users
                SET username = ?, password = ?, profile_image = ?
                WHERE id = ?
            ''', (username, password, profile_image, user_id))
        elif password:
            conn.execute('''
                UPDATE users
                SET username = ?, password = ?
                WHERE id = ?
            ''', (username, password, user_id))
        elif profile_image:
            conn.execute('''
                UPDATE users
                SET username = ?, profile_image = ?
                WHERE id = ?
            ''', (username, profile_image, user_id))
        else:
            conn.execute('''
                UPDATE users
                SET username = ?
                WHERE id = ?
            ''', (username, user_id))
    conn.close()
def get_expenses_by_user_id(user_id):
    conn = sqlite3.connect('personal_finance.db')
    cursor = conn.execute('''
        SELECT date, amount FROM expenses WHERE user_id = ? ORDER BY date
    ''', (user_id,))
    expenses = [{'date': row[0], 'amount': row[1]} for row in cursor.fetchall()]
    conn.close()
    return expenses

def reset_db():
    db_path = 'personal_finance.db'
    if os.path.exists(db_path):
        os.remove(db_path)
    init_db()



# Initialize the database
init_db()
# Ensure the profile_image column exists
add_profile_image_column()
add_description_column()