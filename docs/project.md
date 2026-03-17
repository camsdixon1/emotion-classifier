# Personal Life Logger — Project Overview

## What We're Building

A local, always-on background service that records everything you do on your PC, stores it in a structured SQLite database, and lets you search/filter by person, date, or activity type. Personal memory layer — "what was I working on Tuesday?" or "all my interactions with John last month."

**Scope: PC (Windows) only for now.**

---

## Core Use Cases

- **Daily review** — what did I actually work on today?
- **Person-centric memory** — everything I discussed with a specific contact
- **Audio/call history** — searchable transcripts of calls and meetings
- **Pattern awareness** — where is my time actually going?

---

## Architecture

```
INSTALL & RUN (zero custom code)
────────────────────────────────
Screenpipe (headless binary)   →  screen OCR + audio STT  →  REST API localhost:3030
ActivityWatch                  →  app/window/URL time      →  REST API localhost:5600

CUSTOM PYTHON SERVICE
─────────────────────
keystroke_ingestor.py          →  pynput hooks + win32gui context  →  SQLite
screenshot_ingestor.py         →  mss capture → Claude Vision      →  SQLite
screenpipe_poller.py           →  polls localhost:3030              →  SQLite
activitywatch_poller.py        →  polls localhost:5600              →  SQLite
audio_recorder.py              →  PyAudioWPatch loopback + mic      →  SQLite
                                  + silero-vad + faster-whisper

ENRICH (separate async pass)
─────────────────────────────
enrich.py  →  reads enriched=0  →  Claude Haiku  →  writes summary back

UI (later)
──────────
Obsidian + Dataview plugin for querying, or simple local web page
Review queue — assign transcripts to people
```

**Key principle:** Ingestion and enrichment are fully decoupled. Raw data always gets written first. If Claude's API is down, enrichment stalls but nothing is lost.

---

## V1 Scope (Start Here)

1. **DB setup** — schema below, SQLite WAL mode, single-writer queue pattern
2. **Screenpipe** — install headless binary, poll its REST API for OCR + STT events
3. **ActivityWatch** — install, poll localhost:5600 for app/URL data via `aw-client`
4. **Keystroke ingestor** — pynput + win32gui context + password redaction
5. **Audio recorder** — PyAudioWPatch (loopback + mic) + silero-vad + faster-whisper
6. **Enrich pass** — async Claude summarization of raw events
7. **Review queue v1** — list unreviewed transcripts, type a person's name, done

Everything else comes after real data is flowing.

---

## Database Schema

### Design principles
- Single SQLite file, WAL mode
- **One writer thread only** — all writes go through a `queue.Queue` consumed by a single thread. Never share one connection across threads. Set `PRAGMA busy_timeout=5000` on reader connections.
- Raw data stored first, enrichment happens in a separate pass
- FTS5 for full-text search — built into Python's SQLite, no extra install
- sqlite-vec for semantic/vector search (bolt-on extension, optional)
- DuckDB can query the SQLite file directly for complex analytics — no migration needed

```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- All activity events in one table
CREATE TABLE events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_start    TEXT NOT NULL,      -- ISO8601
    ts_end      TEXT,               -- nullable for point events
    event_type  TEXT NOT NULL,      -- 'window_focus' | 'screenshot' | 'transcript' | 'keystroke_burst' | 'message'
    source      TEXT NOT NULL,      -- 'screenpipe' | 'activitywatch' | 'custom'
    exe         TEXT,
    window_title TEXT,
    url         TEXT,
    raw         TEXT,               -- raw JSON from source, untouched
    enriched    INTEGER DEFAULT 0,  -- 0 = pending, 1 = done
    error       TEXT                -- non-null if enrichment failed
);

-- Keystroke bursts (buffered per-window, not per-key)
CREATE TABLE input_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        INTEGER REFERENCES events(id),
    keycount        INTEGER,
    mouseclick_count INTEGER,
    content         TEXT,           -- buffered typed text
    redacted        INTEGER DEFAULT 0  -- 1 if password field detected
);

-- Screenshots (from mss capture, separate from Screenpipe)
CREATE TABLE screenshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    INTEGER REFERENCES events(id),
    file_path   TEXT,
    ocr_text    TEXT,
    summary     TEXT                -- Claude Vision output
);

-- Audio transcripts (from faster-whisper via our recorder, or from Screenpipe)
CREATE TABLE transcripts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    source      TEXT,               -- 'mic' | 'loopback' | 'meeting' | 'screenpipe'
    audio_path  TEXT,
    raw_text    TEXT,               -- raw whisper output
    summary     TEXT,               -- Claude summary (filled by enrich pass)
    enriched    INTEGER DEFAULT 0,
    error       TEXT
);

-- Messages (email, Slack, etc. — added in later phase)
CREATE TABLE messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    platform    TEXT,               -- 'gmail' | 'slack' | 'sms'
    direction   TEXT,               -- 'inbound' | 'outbound'
    contact_raw TEXT,               -- raw From/To before person resolved
    subject     TEXT,
    body        TEXT,
    thread_id   TEXT,
    enriched    INTEGER DEFAULT 0
);

-- People
CREATE TABLE people (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    notes           TEXT,
    last_seen       TEXT,
    interaction_count INTEGER DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Transcript ↔ People
CREATE TABLE transcript_people (
    transcript_id   INTEGER REFERENCES transcripts(id),
    person_id       INTEGER REFERENCES people(id),
    PRIMARY KEY (transcript_id, person_id)
);

-- Indexes
CREATE INDEX idx_events_ts       ON events(ts_start);
CREATE INDEX idx_events_type     ON events(event_type);
CREATE INDEX idx_events_enriched ON events(enriched);
CREATE INDEX idx_events_exe      ON events(exe);
CREATE INDEX idx_transcripts_enr ON transcripts(enriched);
CREATE INDEX idx_tp_person       ON transcript_people(person_id);

-- FTS5 full-text search (contentless — populate manually on insert)
CREATE VIRTUAL TABLE search_fts USING fts5(
    window_title,
    ocr_text,
    transcript_text,
    message_text,
    content='',
    tokenize='unicode61'
);
```

**Inserting into FTS on write:**
```python
db.execute("""
    INSERT INTO search_fts(rowid, window_title, ocr_text, transcript_text, message_text)
    VALUES (?, ?, ?, ?, ?)
""", (event_id, title, ocr, transcript, message))
```

**Querying FTS:**
```sql
-- Simple
SELECT rowid, rank FROM search_fts WHERE search_fts MATCH 'budget meeting' ORDER BY rank;
-- Phrase
SELECT rowid FROM search_fts WHERE search_fts MATCH '"quarterly review"';
-- Column-specific
SELECT rowid FROM search_fts WHERE search_fts MATCH 'transcript_text: John';
```

**DuckDB analytical queries on top of SQLite (no migration):**
```python
import duckdb
conn = duckdb.connect()
conn.execute("INSTALL sqlite; LOAD sqlite;")
conn.execute("ATTACH 'logger.db' AS logger (TYPE sqlite)")
results = conn.execute("""
    SELECT exe, SUM(CAST(ts_end AS FLOAT) - CAST(ts_start AS FLOAT)) / 3600 as hours
    FROM logger.events
    WHERE ts_start > (datetime('now', '-7 days'))
    GROUP BY exe ORDER BY hours DESC
""").fetchall()
```

---

## Tool Decisions (Research-Backed)

### Screen + Audio: Use Screenpipe headless

- Run the MIT-licensed Rust binary headlessly — no desktop app, no $400 license needed
- Captures screen OCR + audio STT automatically, stores in its own SQLite
- Query from Python via REST: `GET http://localhost:3030/search?contentType=ocr&startTime=...`
- **No Python SDK** — use `httpx` or `requests` to poll the REST API
- API is not versioned — watch for field changes on upgrades

### App/Window/URL: ActivityWatch + aw-client

```python
pip install aw-client
```
```python
from aw_client import ActivityWatchClient
client = ActivityWatchClient("lifelogger")
events = client.get_events("aw-watcher-window_hostname", limit=100)
```
- Browser URL requires the `aw-watcher-web` browser extension to be installed

### Screenshots: mss (fastest Python option)

```python
pip install mss
```
- ~3ms per full-screen capture vs ~100ms for PIL.ImageGrab
- Thread-safe, no runtime dependencies
- Use for our own periodic screenshot pipeline separate from Screenpipe

### Keystroke + Mouse: pynput (critical gotcha)

```python
pip install pynput pywin32 psutil
```

**The most important pynput rule:** The OS hook callback runs on the Windows input thread. **It must never block.** Dispatching to a queue is mandatory — any slow work (disk write, API call) done directly in the callback will stall system-wide input.

```python
from pynput import keyboard
import queue

q = queue.Queue()

def on_press(key):
    try:
        q.put_nowait(key)   # never block here
    except Exception:
        pass                 # swallow silently — never let the callback raise
```

A separate worker thread drains the queue, combines with `win32gui.GetForegroundWindow()` context, and writes to DB.

**Getting URL from browser:** Use `uiautomation` package (not `comtypes` directly — it's a higher-level wrapper). The Chrome/Edge address bar is a named UIAutomation element.
```python
pip install uiautomation
```

**Password field detection (two layers):**
```python
import win32gui, win32con, uiautomation as auto

def is_sensitive(hwnd):
    # Layer 1: Win32 ES_PASSWORD style flag (works for native apps)
    style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
    if style & 0x0020:
        return True
    # Layer 2: UIA IsPassword property (works for browsers + modern apps)
    focused = auto.GetFocusedControl()
    if getattr(focused, 'IsPassword', False):
        return True
    return False
```
Also maintain a process name blacklist: `1Password.exe`, `Bitwarden.exe`, `KeePass.exe`.

### Audio: PyAudioWPatch (NOT sounddevice)

**sounddevice does not support WASAPI loopback.** This is a known limitation — it wraps stock PortAudio which lacks loopback. Use PyAudioWPatch instead:

```python
pip install PyAudioWPatch
```

PyAudioWPatch is a drop-in PyAudio replacement that ships a PortAudio build with WASAPI loopback support. Windows-only, Python 3.7–3.13, wheels available.

Finding the loopback device:
```python
import pyaudiowpatch as pyaudio

with pyaudio.PyAudio() as p:
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev["name"] == default_speakers["name"] + " [Loopback]":
            loopback_device = dev
            break
```

Use PyAudioWPatch for both mic (input) and loopback (system audio). Two separate streams, merged for transcription.

### Voice Activity Detection: silero-vad

```python
pip install silero-vad
```

- ~1.8MB model, processes 32ms chunks in <1ms on CPU
- Feed 512 samples at a time (at 16kHz), get back speech start/end events
- **Buffer audio only during detected speech segments** — send completed utterances to Whisper
- Call `vad_iterator.reset_states()` between separate audio streams
- Reduces storage by ~80% vs recording continuously

### Transcription: faster-whisper

```python
pip install faster-whisper
```

| Model | RAM (INT8 CPU) | Use for |
|---|---|---|
| tiny | ~300 MB | Testing only |
| base | ~500 MB | Quick notes, high CPU budget |
| medium | ~2 GB | **Sweet spot** — good accuracy, practical on CPU |
| large-v3 | ~4 GB | Best accuracy, needs 8GB+ RAM |

```python
from faster_whisper import WhisperModel
model = WhisperModel("medium", device="cpu", compute_type="int8")
segments, info = model.transcribe("chunk.wav", beam_size=5)
```

No FFmpeg needed. No GPU required (CPU INT8 mode).

### Meeting Detection: psutil

```python
import psutil

MEETING_PROCESSES = {"Zoom.exe", "Teams.exe", "ms-teams.exe", "CiscoCollabHost.exe", "slack.exe"}

def is_in_meeting():
    running = {p.info['name'] for p in psutil.process_iter(['name'])}
    return bool(running & MEETING_PROCESSES)
```

When a meeting process is detected: start recording both audio streams automatically.

### Search Layers

| Need | Tool | Notes |
|---|---|---|
| Full-text (keyword) | SQLite FTS5 | Built in, zero setup, fast |
| Vector/semantic | sqlite-vec | Bolt-on extension, brute-force KNN, fine under 1M vectors |
| Analytics/aggregations | DuckDB | Queries the SQLite file directly, no migration |
| Hybrid search | LanceDB | If sqlite-vec isn't enough — separate store with HNSW index |

---

## What Already Exists vs What to Build

| Capability | Approach |
|---|---|
| Screen OCR + audio STT | **Install Screenpipe headless**, poll REST API |
| App/window/URL time | **Install ActivityWatch**, poll via aw-client |
| Browser URL extraction | **Install aw-watcher-web** extension (ActivityWatch handles it) |
| Keystroke + mouse context | **Build** — pynput + win32gui queue pattern |
| Audio loopback capture | **Build** — PyAudioWPatch + silero-vad + faster-whisper |
| Password redaction | **Build** — Win32 ES_PASSWORD + UIA IsPassword |
| Meeting auto-trigger | **Build** — psutil process watcher (trivial) |
| Full-text search | **SQLite FTS5** — set up at schema creation time |
| Semantic search | **sqlite-vec** — pip install + schema addition |

---

## Project Structure

```
lifelogger/
  db.py                     # SQLite setup, WAL, single-writer queue, FTS helpers
  ingest/
    screenpipe_poller.py    # polls localhost:3030, writes to events + transcripts
    activitywatch_poller.py # polls localhost:5600 via aw-client, writes to events
    keystroke_ingestor.py   # pynput + win32gui context + password redaction
    audio_recorder.py       # PyAudioWPatch + silero-vad + faster-whisper
    screenshot_ingestor.py  # mss + Claude Vision (optional, Screenpipe may cover this)
  enrich.py                 # async pass: reads enriched=0, calls Claude, writes back
  scheduler.py              # APScheduler with job failure logging
  main.py                   # boots everything, handles shutdown
```

---

## Deployment — Local, Always-On (Windows)

```
Windows startup
  └── NSSM (Non-Sucking Service Manager) wraps: python main.py
        ├── Screenpipe headless binary (separate service)
        ├── ActivityWatch aw-server (separate service)
        └── lifelogger Python service
              ├── screenpipe_poller   (every 60s)
              ├── activitywatch_poller (every 30s)
              ├── keystroke_ingestor  (continuous, event-driven)
              ├── audio_recorder      (continuous with VAD)
              └── enrich.py           (every 10 min)
```

All data in: `C:\Users\{user}\lifelogger\logger.db` + audio files in `C:\Users\{user}\lifelogger\audio\`

---

## Honest Hard Parts

1. **pynput callback blocking** — silent disaster. Must queue immediately, never do work in callback.
2. **sounddevice doesn't do loopback** — use PyAudioWPatch. Easy fix once you know.
3. **Browser URL via UIA** — works but requires target window to be visible/partially active. Edge cases exist.
4. **Password detection coverage** — Win32 flag + UIA IsPassword covers most cases but not all. Maintain a process blacklist as fallback.
5. **Screenpipe API changes** — not versioned, fields shift between releases. Pin the binary version.
6. **Data volume** — Screenpipe generates ~5–10 GB/month of video chunks. Define retention before it fills a drive.
7. **faster-whisper on CPU** — `medium` model is ~2x real-time on a modern CPU. Fine for post-processing meeting recordings, too slow for live continuous transcription. Use VAD to only transcribe actual speech.

---

## Build Order

```
1. db.py                      — schema, WAL, single-writer queue, FTS5 setup
2. screenpipe_poller.py        — get Screenpipe running + poll its REST API
3. activitywatch_poller.py     — get AW running + poll via aw-client
4. keystroke_ingestor.py       — pynput queue pattern + win32gui + password redaction
5. audio_recorder.py           — PyAudioWPatch + silero-vad + faster-whisper pipeline
6. enrich.py                   — async Claude summarization pass
7. review_queue (simple HTML)  — list transcripts, assign person, mark done
8. scheduler.py / main.py      — boots everything, APScheduler with error logging
```
