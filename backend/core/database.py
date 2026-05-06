"""
core/database.py
SQLite connection helper and schema initialisation.
"""

import sqlite3
from config import DB_PATH


def get_conn() -> sqlite3.Connection:
    """Return a new SQLite connection with row_factory set."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't already exist."""
    conn = get_conn()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id   TEXT UNIQUE NOT NULL,
            name         TEXT NOT NULL,
            roll_number  TEXT,
            class_name   TEXT,
            email        TEXT,
            photo_path   TEXT,
            enrolled_at  TEXT,
            face_samples INTEGER DEFAULT 0
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id   TEXT NOT NULL,
            student_name TEXT NOT NULL,
            date         TEXT NOT NULL,
            time         TEXT NOT NULL,
            confidence   REAL,
            status       TEXT DEFAULT 'Present',
            UNIQUE(student_id, date)
        )
    ''')

    conn.commit()
    conn.close()
