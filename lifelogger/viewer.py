"""
Simple viewer to see what's in the database.
Run this anytime to see recent activity.

Usage:
    python viewer.py                    # show last 30 min of everything
    python viewer.py --hours 4          # last 4 hours
    python viewer.py --search "budget"  # search all tables
"""

import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path.home() / "lifelogger" / "logger.db"


def connect():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def show_recent(hours=0.5):
    conn = connect()
    since = (datetime.now() - timedelta(hours=hours)).isoformat()

    print(f"\n{'='*70}")
    print(f"  LIFELOGGER — Last {hours} hours (since {since[:19]})")
    print(f"{'='*70}")

    # Keystrokes
    rows = conn.execute(
        "SELECT ts, exe, window, text, key_count, redacted "
        "FROM keystrokes WHERE ts > ? ORDER BY ts", (since,)
    ).fetchall()
    print(f"\n--- KEYSTROKES ({len(rows)} bursts) ---")
    for r in rows:
        ts = r["ts"][:19]
        exe = (r["exe"] or "")[:20]
        win = (r["window"] or "")[:40]
        if r["redacted"]:
            text = f"[REDACTED {r['key_count']} keys]"
        else:
            text = (r["text"] or "")[:60].replace("\n", "⏎")
        print(f"  {ts}  {exe:<20}  {win:<40}  {text}")

    # Clicks
    rows = conn.execute(
        "SELECT ts, exe, window, x, y, button "
        "FROM clicks WHERE ts > ? ORDER BY ts", (since,)
    ).fetchall()
    print(f"\n--- CLICKS ({len(rows)} clicks) ---")
    for r in rows:
        ts = r["ts"][:19]
        exe = (r["exe"] or "")[:20]
        win = (r["window"] or "")[:40]
        pos = f"({r['x']},{r['y']})"
        print(f"  {ts}  {exe:<20}  {win:<40}  {r['button']:<6} {pos}")

    # Transcripts
    rows = conn.execute(
        "SELECT ts_start, ts_end, source, text "
        "FROM transcripts WHERE ts_start > ? ORDER BY ts_start", (since,)
    ).fetchall()
    print(f"\n--- TRANSCRIPTS ({len(rows)} segments) ---")
    for r in rows:
        ts = r["ts_start"][:19]
        src = r["source"] or "?"
        text = (r["text"] or "")[:100]
        print(f"  {ts}  [{src:<8}]  {text}")

    # Stats
    print(f"\n--- STATS ---")
    total_keys = conn.execute(
        "SELECT COALESCE(SUM(key_count),0) FROM keystrokes WHERE ts > ?", (since,)
    ).fetchone()[0]
    total_clicks = conn.execute(
        "SELECT COUNT(*) FROM clicks WHERE ts > ?", (since,)
    ).fetchone()[0]
    total_words = conn.execute(
        "SELECT COALESCE(SUM(LENGTH(text) - LENGTH(REPLACE(text,' ','')) + 1), 0) "
        "FROM transcripts WHERE ts_start > ?", (since,)
    ).fetchone()[0]
    print(f"  Keystrokes: {total_keys}")
    print(f"  Clicks:     {total_clicks}")
    print(f"  Words transcribed: {total_words}")

    conn.close()


def search(keyword):
    conn = connect()
    print(f"\n{'='*70}")
    print(f"  SEARCH: '{keyword}'")
    print(f"{'='*70}")

    rows = conn.execute(
        "SELECT ts, exe, window, text FROM keystrokes "
        "WHERE text LIKE ? OR window LIKE ? ORDER BY ts DESC LIMIT 20",
        (f"%{keyword}%", f"%{keyword}%")
    ).fetchall()
    print(f"\n--- Keystrokes matching '{keyword}' ({len(rows)} results) ---")
    for r in rows:
        ts = r["ts"][:19]
        text = (r["text"] or "")[:80].replace("\n", "⏎")
        print(f"  {ts}  [{r['exe']}]  {text}")

    rows = conn.execute(
        "SELECT ts_start, source, text FROM transcripts "
        "WHERE text LIKE ? ORDER BY ts_start DESC LIMIT 20",
        (f"%{keyword}%",)
    ).fetchall()
    print(f"\n--- Transcripts matching '{keyword}' ({len(rows)} results) ---")
    for r in rows:
        ts = r["ts_start"][:19]
        text = (r["text"] or "")[:100]
        print(f"  {ts}  [{r['source']}]  {text}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View lifelogger data")
    parser.add_argument("--hours", type=float, default=0.5, help="Hours of history to show")
    parser.add_argument("--search", type=str, help="Search for a keyword")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"No database found at {DB_PATH}")
        print("Run main.py first to start collecting data.")
        exit(1)

    if args.search:
        search(args.search)
    else:
        show_recent(args.hours)
