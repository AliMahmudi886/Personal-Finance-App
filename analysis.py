import pandas as pd
import sqlite3
from database import create_connection

def get_expenses(user_id):
    conn = create_connection()
    query = 'SELECT * FROM expenses WHERE user_id = ?'
    df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()
    return df

def visualize_expenses(user_id):
    df = get_expenses(user_id)
    df['date'] = pd.to_datetime(df['date'])
    df.groupby(df['date'].dt.to_period('M')).sum().plot(kind='bar')
    plt.show()
