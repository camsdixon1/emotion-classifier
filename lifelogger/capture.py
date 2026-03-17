"""
Keystroke + mouse click capture using pynput.

CRITICAL: pynput callbacks run on the Windows input hook thread.
They must NEVER block. All we do in the callback is queue.put_nowait().
A separate worker thread drains the queue, gets window context, and writes to DB.
"""

import queue
import threading
import time
import sys
from datetime import datetime

# Buffer keystrokes per-window for this many seconds before flushing
FLUSH_INTERVAL = 3.0


def _get_window_context():
    """Get the active window title and exe name. Windows-only via win32gui."""
    try:
        import win32gui
        import win32process
        import psutil

        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        try:
            proc = psutil.Process(pid)
            exe = proc.name()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            exe = "unknown"
        return exe, title
    except ImportError:
        # Not on Windows or missing pywin32 — return placeholder
        return "unknown", "unknown"


def _is_password_field():
    """Check if the currently focused field is a password input."""
    try:
        import win32gui
        import win32con

        hwnd = win32gui.GetFocus()
        if hwnd:
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
            if style & 0x0020:  # ES_PASSWORD
                return True
    except Exception:
        pass

    try:
        import uiautomation as auto
        focused = auto.GetFocusedControl()
        if getattr(focused, "IsPassword", False):
            return True
    except Exception:
        pass

    return False


# Process names where we never log keystrokes
SENSITIVE_PROCESSES = {
    "1password.exe", "bitwarden.exe", "keepass.exe", "keepassxc.exe",
    "lastpass.exe", "dashlane.exe",
}


class InputCapture:
    def __init__(self, db):
        self.db = db
        self._key_queue = queue.Queue()
        self._click_queue = queue.Queue()

        # Keystroke buffer: accumulate text per (exe, window) before flushing
        self._buffer_lock = threading.Lock()
        self._buffer_text = ""
        self._buffer_count = 0
        self._buffer_exe = None
        self._buffer_window = None
        self._buffer_redacted = False
        self._last_flush = time.time()

        self._running = True

    def start(self):
        """Start listeners + worker threads."""
        threading.Thread(target=self._key_worker, daemon=True, name="key-worker").start()
        threading.Thread(target=self._click_worker, daemon=True, name="click-worker").start()
        threading.Thread(target=self._flush_timer, daemon=True, name="flush-timer").start()

        from pynput import keyboard, mouse
        self._key_listener = keyboard.Listener(on_press=self._on_key_press)
        self._mouse_listener = mouse.Listener(on_click=self._on_mouse_click)
        self._key_listener.start()
        self._mouse_listener.start()
        print("[capture] Listening for keystrokes + clicks")

    def stop(self):
        self._running = False
        self._flush_buffer()
        self._key_listener.stop()
        self._mouse_listener.stop()

    # --- Callbacks (MUST NOT BLOCK) ---

    def _on_key_press(self, key):
        try:
            self._key_queue.put_nowait(key)
        except Exception:
            pass

    def _on_mouse_click(self, x, y, button, pressed):
        if pressed:  # only log press, not release
            try:
                self._click_queue.put_nowait((x, y, button))
            except Exception:
                pass

    # --- Workers (run in their own threads, can do slow work) ---

    def _key_worker(self):
        while self._running:
            try:
                key = self._key_queue.get(timeout=1)
            except queue.Empty:
                continue

            exe, window = _get_window_context()

            # Check sensitive process
            if exe.lower() in SENSITIVE_PROCESSES:
                redacted = True
                char = None
            elif _is_password_field():
                redacted = True
                char = None
            else:
                redacted = False
                # Convert key to string
                try:
                    char = key.char  # regular character
                except AttributeError:
                    # Special key
                    if key == keyboard.Key.space:
                        char = " "
                    elif key == keyboard.Key.enter:
                        char = "\n"
                    elif key == keyboard.Key.tab:
                        char = "\t"
                    elif key == keyboard.Key.backspace:
                        char = "[BS]"
                    else:
                        char = f"[{key.name}]"

            with self._buffer_lock:
                # If window changed, flush the old buffer
                if exe != self._buffer_exe or window != self._buffer_window:
                    self._flush_buffer_locked()
                    self._buffer_exe = exe
                    self._buffer_window = window
                    self._buffer_redacted = redacted

                if redacted:
                    self._buffer_redacted = True
                    self._buffer_count += 1
                elif char:
                    self._buffer_text += char
                    self._buffer_count += 1

    def _click_worker(self):
        while self._running:
            try:
                x, y, button = self._click_queue.get(timeout=1)
            except queue.Empty:
                continue

            exe, window = _get_window_context()
            btn_name = button.name if hasattr(button, "name") else str(button)
            self.db.log_click(exe, window, x, y, btn_name)

    def _flush_timer(self):
        """Periodically flush the keystroke buffer."""
        while self._running:
            time.sleep(FLUSH_INTERVAL)
            self._flush_buffer()

    def _flush_buffer(self):
        with self._buffer_lock:
            self._flush_buffer_locked()

    def _flush_buffer_locked(self):
        if self._buffer_count == 0:
            return
        self.db.log_keystroke(
            exe=self._buffer_exe,
            window=self._buffer_window,
            text=None if self._buffer_redacted else self._buffer_text,
            key_count=self._buffer_count,
            redacted=self._buffer_redacted,
        )
        self._buffer_text = ""
        self._buffer_count = 0
        self._buffer_redacted = False
