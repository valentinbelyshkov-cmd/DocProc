import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager
from config import DB_PATH

def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with get_db() as db:
        db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                total_pages INTEGER DEFAULT 0,
                processed_pages INTEGER DEFAULT 0,
                detect_seal INTEGER DEFAULT 0,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(id),
                page_num INTEGER NOT NULL,
                markdown TEXT NOT NULL,
                result_json TEXT,
                created_at REAL NOT NULL,
                UNIQUE(job_id, page_num)
            );
        """)
        # Migration: ensure result_json exists
        try:
            db.execute("SELECT result_json FROM pages LIMIT 1")
        except sqlite3.OperationalError:
            db.execute("ALTER TABLE pages ADD COLUMN result_json TEXT")

        # Migration: ensure detect_seal exists
        try:
            db.execute("SELECT detect_seal FROM jobs LIMIT 1")
        except sqlite3.OperationalError:
            db.execute("ALTER TABLE jobs ADD COLUMN detect_seal INTEGER DEFAULT 0")

        now = time.time()
        stale = db.execute("SELECT id FROM jobs WHERE status = 'processing'").fetchall()
        for row in stale:
            db.execute("DELETE FROM pages WHERE job_id = ?", (row["id"],))
        db.execute(
            "UPDATE jobs SET status = 'queued', processed_pages = 0, updated_at = ? WHERE status = 'processing'",
            (now,),
        )

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
