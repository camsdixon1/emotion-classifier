"""
Single-writer SQLite database for the lifelogger.

All writes go through a queue consumed by one thread.
Readers can open their own connections freely (WAL mode allows concurrent reads).
"""

import sqlite3
import queue
import threading
import os
from datetime import datetime
from pathlib import Path

# Default DB location
DEFAULT_DB_DIR = Path.home() / "lifelogger"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "logger.db"

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS keystrokes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    exe         TEXT,
    window      TEXT,
    text        TEXT,
    key_count   INTEGER DEFAULT 1,
    redacted    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS clicks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    exe         TEXT,
    window      TEXT,
    x           INTEGER,
    y           INTEGER,
    button      TEXT
);

CREATE TABLE IF NOT EXISTS transcripts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_start    TEXT NOT NULL,
    ts_end      TEXT,
    source      TEXT,
    text        TEXT NOT NULL,
    audio_path  TEXT
);

CREATE INDEX IF NOT EXISTS idx_k_ts ON keystrokes(ts);
CREATE INDEX IF NOT EXISTS idx_c_ts ON clicks(ts);
CREATE INDEX IF NOT EXISTS idx_t_ts ON transcripts(ts_start);
CREATE INDEX IF NOT EXISTS idx_k_exe ON keystrokes(exe);
CREATE INDEX IF NOT EXISTS idx_c_exe ON clicks(exe);
"""

# Sentinel to tell the writer thread to stop
_STOP = object()


class DB:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._queue = queue.Queue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="db-writer"
        )
        self._init_schema()
        self._writer_thread.start()

    def _init_schema(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript(SCHEMA)
        conn.close()

    def _writer_loop(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        while True:
            item = self._queue.get()
            if item is _STOP:
                conn.close()
                return
            sql, params = item
            try:
                conn.execute(sql, params)
                conn.commit()
            except Exception as e:
                print(f"[db] write error: {e}")

    # --- Public write methods (called from any thread) ---

    def log_keystroke(self, exe, window, text, key_count=1, redacted=False):
        ts = datetime.now().isoformat()
        self._queue.put((
            "INSERT INTO keystrokes(ts, exe, window, text, key_count, redacted) "
            "VALUES(?,?,?,?,?,?)",
            (ts, exe, window, text, key_count, 1 if redacted else 0),
        ))

    def log_click(self, exe, window, x, y, button):
        ts = datetime.now().isoformat()
        self._queue.put((
            "INSERT INTO clicks(ts, exe, window, x, y, button) VALUES(?,?,?,?,?,?)",
            (ts, exe, window, x, y, button),
        ))

    def log_transcript(self, ts_start, ts_end, source, text, audio_path=None):
        self._queue.put((
            "INSERT INTO transcripts(ts_start, ts_end, source, text, audio_path) "
            "VALUES(?,?,?,?,?)",
            (ts_start, ts_end, source, text, audio_path),
        ))

    # --- Read helpers (open a new connection each time, safe in WAL) ---

    def _read_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def recent_keystrokes(self, limit=20):
        conn = self._read_conn()
        rows = conn.execute(
            "SELECT * FROM keystrokes ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return rows

    def recent_clicks(self, limit=20):
        conn = self._read_conn()
        rows = conn.execute(
            "SELECT * FROM clicks ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return rows

    def recent_transcripts(self, limit=10):
        conn = self._read_conn()
        rows = conn.execute(
            "SELECT * FROM transcripts ORDER BY ts_start DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return rows

    def search(self, keyword, table="keystrokes", limit=50):
        """Simple LIKE search across text fields."""
        conn = self._read_conn()
        if table == "keystrokes":
            rows = conn.execute(
                "SELECT * FROM keystrokes WHERE text LIKE ? OR window LIKE ? "
                "ORDER BY ts DESC LIMIT ?",
                (f"%{keyword}%", f"%{keyword}%", limit),
            ).fetchall()
        elif table == "clicks":
            rows = conn.execute(
                "SELECT * FROM clicks WHERE window LIKE ? ORDER BY ts DESC LIMIT ?",
                (f"%{keyword}%", limit),
            ).fetchall()
        elif table == "transcripts":
            rows = conn.execute(
                "SELECT * FROM transcripts WHERE text LIKE ? "
                "ORDER BY ts_start DESC LIMIT ?",
                (f"%{keyword}%", limit),
            ).fetchall()
        else:
            rows = []
        conn.close()
        return rows

    def stop(self):
        self._queue.put(_STOP)
        self._writer_thread.join(timeout=5)


if __name__ == "__main__":
    # Quick self-test
    db = DB(db_path="test_logger.db")
    db.log_keystroke("Code.exe", "main.py - VSCode", "def hello():", 12)
    db.log_keystroke("chrome.exe", "Google - Chrome", "how to cook pasta", 18)
    db.log_click("chrome.exe", "Google - Chrome", 450, 320, "left")
    db.log_click("explorer.exe", "Downloads", 100, 200, "right")
    db.log_transcript(
        "2026-03-17T14:00:00", "2026-03-17T14:02:30", "mic",
        "Hey Sarah, can you send me the Q1 report by Friday?"
    )
    db.log_transcript(
        "2026-03-17T14:02:31", "2026-03-17T14:03:00", "loopback",
        "Sure, I'll have it ready by Thursday actually."
    )

    import time
    time.sleep(0.5)  # let writer thread flush

    print("=== RECENT KEYSTROKES ===")
    for row in db.recent_keystrokes():
        print(dict(row))

    print("\n=== RECENT CLICKS ===")
    for row in db.recent_clicks():
        print(dict(row))

    print("\n=== RECENT TRANSCRIPTS ===")
    for row in db.recent_transcripts():
        print(dict(row))

    print("\n=== SEARCH: 'Sarah' in transcripts ===")
    for row in db.search("Sarah", table="transcripts"):
        print(dict(row))

    db.stop()
    os.remove("test_logger.db")
    print("\nSelf-test passed.")
