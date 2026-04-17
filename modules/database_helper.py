import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = 'database/history.db'

def init_db():

    os.makedirs('database/captured_scans', exist_ok=True)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Tambahkan kolom image_path
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_used TEXT,
            prediction TEXT,
            confidence REAL,
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_scan(model_used, prediction, confidence, image_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO scan_history (timestamp, model_used, prediction, confidence, image_path)
        VALUES (?, ?, ?, ?, ?)
    ''', (current_time, model_used, prediction, confidence, image_path))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM scan_history ORDER BY id DESC", conn)
    conn.close()
    return df

def delete_session(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scan_history WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()