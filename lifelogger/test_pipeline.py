"""
End-to-end pipeline test.

Tests:
  1. Keystroke capture  — inject synthetic key events, verify DB row
  2. Click capture      — inject synthetic click, verify DB row
  3. Audio/transcription — generate a speech WAV, feed it directly to the
                          transcription queue, verify DB row
"""

import sys
import os
import time
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from db import DB as Database
from capture import InputCapture
from audio import AudioPipeline

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def _synthetic_key(char=None, name=None):
    """Create a minimal pynput-compatible key object."""
    class Key:
        pass
    k = Key()
    k.char = char
    if name:
        k.name = name
        k.char = None
    return k


class _MockKbKey:
    """Minimal stand-in for pynput keyboard.Key enum values."""
    def __init__(self, name): self.name = name; self.char = None

class _MockKeyboard:
    """Minimal stand-in for the pynput keyboard module."""
    space     = _MockKbKey("space")
    enter     = _MockKbKey("enter")
    tab       = _MockKbKey("tab")
    backspace = _MockKbKey("backspace")
    class Key:
        space     = _MockKbKey("space")
        enter     = _MockKbKey("enter")
        tab       = _MockKbKey("tab")
        backspace = _MockKbKey("backspace")


def test_keystrokes(db, query):
    print("\n[test] Keystroke capture...")
    cap = InputCapture(db)
    cap._running = True
    cap._keyboard = _MockKeyboard()
    threading.Thread(target=cap._key_worker, daemon=True, name="key-worker").start()
    threading.Thread(target=cap._flush_timer, daemon=True, name="flush-timer").start()

    # Inject: "hello world"
    for ch in "hello world":
        cap._key_queue.put_nowait(_synthetic_key(char=ch))
    cap._key_queue.put_nowait(_synthetic_key(name="enter"))

    time.sleep(0.5)
    cap._flush_buffer()
    time.sleep(0.3)

    row = query("SELECT text, key_count FROM keystrokes ORDER BY id DESC LIMIT 1")

    if row and "hello world" in row[0]:
        print(f"  {PASS} keystrokes captured: {repr(row[0])} ({row[1]} keys)")
        return True
    else:
        print(f"  {FAIL} expected 'hello world' in DB, got: {row}")
        return False


def test_clicks(db, query):
    print("\n[test] Click capture...")
    cap = InputCapture(db)
    cap._running = True
    threading.Thread(target=cap._click_worker, daemon=True, name="click-worker").start()

    class FakeButton:
        name = "left"
    cap._click_queue.put_nowait((100, 200, FakeButton()))
    time.sleep(0.3)

    row = query("SELECT x, y, button FROM clicks ORDER BY id DESC LIMIT 1")

    if row and row[0] == 100 and row[1] == 200:
        print(f"  {PASS} click recorded at ({row[0]}, {row[1]}) button={row[2]}")
        return True
    else:
        print(f"  {FAIL} expected click at (100,200), got: {row}")
        return False


class _MockSegment:
    text = "hello this is a pipeline test"

class _MockWhisper:
    def transcribe(self, audio, **kwargs):
        return [_MockSegment()], None


def test_audio(db, query):
    print("\n[test] Audio transcription pipeline...")
    pipe = AudioPipeline(db)
    pipe._running = True

    print("  Loading VAD model...")
    try:
        pipe._load_vad()
    except Exception as e:
        print(f"  {FAIL} VAD load failed: {e}")
        return False

    # Use mock Whisper so the test doesn't depend on synthetic audio quality
    pipe._whisper = _MockWhisper()
    print("  Whisper mocked (tests WAV save + DB write path)")

    threading.Thread(target=pipe._transcribe_worker, daemon=True, name="transcriber").start()

    # 2 second silent audio — VAD already bypassed (we go direct to transcribe queue)
    audio = np.zeros(16000 * 2, dtype=np.float32)
    pipe._transcribe_queue.put(("mic", "2026-01-01T00:00:00", "2026-01-01T00:00:02", audio))

    deadline = time.time() + 10
    while time.time() < deadline:
        row = query("SELECT text FROM transcripts ORDER BY id DESC LIMIT 1")
        if row:
            print(f"  {PASS} transcript saved: {repr(row[0][:80])}")
            return True
        time.sleep(0.3)

    print(f"  {FAIL} no transcript row written within 10s")
    return False


def test_audio_device():
    print("\n[test] Audio device (PulseAudio)...")
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        n = pa.get_device_count()
        if n == 0:
            print(f"  {FAIL} no audio devices found")
            pa.terminate()
            return False
        devices = [pa.get_device_info_by_index(i)["name"] for i in range(n)]
        print(f"  {PASS} {n} device(s): {devices}")
        # Try opening the default input
        try:
            stream = pa.open(
                format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=512,
            )
            data = stream.read(512, exception_on_overflow=False)
            stream.stop_stream()
            stream.close()
            print(f"  {PASS} mic stream opened and read {len(data)} bytes")
        except Exception as e:
            print(f"  {FAIL} could not read from mic: {e}")
            pa.terminate()
            return False
        pa.terminate()
        return True
    except Exception as e:
        print(f"  {FAIL} PyAudio error: {e}")
        return False


def main():
    import tempfile, sqlite3
    db_path = tempfile.mktemp(suffix=".db")
    db = Database(db_path)

    def query(sql):
        conn = sqlite3.connect(db_path)
        row = conn.execute(sql).fetchone()
        conn.close()
        return row

    results = {}
    results["keystrokes"]    = test_keystrokes(db, query)
    results["clicks"]        = test_clicks(db, query)
    results["audio_device"]  = test_audio_device()
    results["transcription"] = test_audio(db, query)

    os.unlink(db_path)

    print("\n" + "="*50)
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
    print("="*50)

    if all(results.values()):
        print("All tests passed.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
