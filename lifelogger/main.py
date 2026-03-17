"""
Lifelogger — main entry point.

Boots the keystroke/click capture and audio pipeline, runs until Ctrl+C.
"""

import signal
import sys
import time

from db import DB
from capture import InputCapture
from audio import AudioPipeline


def main():
    print("=" * 50)
    print("  Lifelogger starting...")
    print("=" * 50)

    db = DB()
    print(f"[main] Database: {db.db_path}")

    # Start input capture (keystrokes + clicks)
    capture = InputCapture(db)
    try:
        capture.start()
    except Exception as e:
        print(f"[main] Input capture failed to start: {e}")
        print("[main] Continuing without keystroke/click capture")
        capture = None

    # Start audio pipeline (mic + loopback → VAD → whisper → DB)
    audio = AudioPipeline(db)
    try:
        audio.start()
    except Exception as e:
        print(f"[main] Audio pipeline failed to start: {e}")
        print("[main] Continuing without audio (install requirements to enable)")

    print()
    print("Lifelogger is running. Press Ctrl+C to stop.")
    print(f"Data stored in: {db.db_path}")
    print()

    # Handle graceful shutdown
    def shutdown(sig, frame):
        print("\n[main] Shutting down...")
        if capture:
            capture.stop()
        audio.stop()
        db.stop()
        print("[main] Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep main thread alive
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
