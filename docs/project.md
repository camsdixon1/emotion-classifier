# Personal Life Logger — Project Overview

## What We're Building

A local, always-on background service that records everything you do on PC and Android, stores it in a structured SQLite database, and lets you search/filter your life by person, date, or activity type. Think of it as a personal memory layer — "what was I working on Tuesday?" or "all my interactions with John last month."

---

## Core Use Cases

- **Daily review** — what did I actually work on today?
- **Person-centric memory** — everything I discussed with a specific contact
- **Audio/call history** — searchable transcripts of calls and meetings
- **Note integration** — link notes written in Obsidian/markdown to people and events
- **Pattern awareness** — where is my time actually going?

---

## Architecture

```
CAPTURE                         INGEST                          STORE
───────                         ──────                          ─────
Screenshot daemon (every 5min)  →  screenshot_ingestor.py   →  events table
ActivityWatch (app/window)      →  activitywatch_ingestor.py →  events table
Audio drop folder               →  audio_ingestor.py        →  transcripts table
Obsidian vault (file watcher)   →  vault_ingestor.py        →  notes table (later)

ENRICH (separate pass, async)
─────────────────────────────
enrich.py  →  reads enriched=0  →  calls Claude Haiku Vision  →  writes summary back

UI (later)
──────────
Simple web UI or Obsidian + Dataview for querying
Review queue — assign transcripts to people
```

**Key principle:** Ingestion and enrichment are fully decoupled. If Claude's API is down, ingestion keeps running. Nothing is lost.

---

## V1 Scope (Start Here)

1. **DB setup** — 3 tables, SQLite WAL mode, single writer
2. **Screenshot ingestor** — capture every 5 min, write raw to `events`
3. **ActivityWatch ingestor** — poll localhost:5600, write raw JSON to `events`
4. **Audio ingestor** — watch a folder, run faster-whisper, write to `transcripts`
5. **Enrich pass** — separate process, reads `enriched=0`, calls Claude, writes summary
6. **Review queue v1** — list unreviewed transcripts, type a person's name, link and mark done

Everything else comes after real data is flowing.

---

## Database Schema (V1 — minimal, intentionally sparse)

```sql
CREATE TABLE events (
  id        INTEGER PRIMARY KEY,
  ts        TIMESTAMP NOT NULL,
  source    TEXT NOT NULL,       -- 'screenshot' | 'activitywatch'
  raw       TEXT NOT NULL,       -- raw JSON/data from source, untouched
  enriched  INTEGER DEFAULT 0,   -- 0 = needs processing, 1 = done
  error     TEXT                 -- non-null if enrichment failed
);

CREATE TABLE transcripts (
  id         INTEGER PRIMARY KEY,
  ts         TIMESTAMP NOT NULL,
  source     TEXT,               -- 'call' | 'meeting' | 'voice_note'
  audio_path TEXT,
  raw_text   TEXT,               -- raw whisper output, nothing more
  summary    TEXT,               -- filled in by enrich pass
  enriched   INTEGER DEFAULT 0,
  error      TEXT
);

CREATE TABLE people (
  id    INTEGER PRIMARY KEY,
  name  TEXT NOT NULL,
  notes TEXT
);

-- Links transcripts to people (added after review queue)
CREATE TABLE transcript_people (
  transcript_id INTEGER REFERENCES transcripts(id),
  person_id     INTEGER REFERENCES people(id),
  PRIMARY KEY (transcript_id, person_id)
);

-- Indexes
CREATE INDEX idx_events_ts        ON events(ts);
CREATE INDEX idx_events_enriched  ON events(enriched);
CREATE INDEX idx_transcripts_enr  ON transcripts(enriched);
```

Schema evolves only when real data shows a column is needed.

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Screenshots | `mss` + `Pillow` | Python, cross-platform |
| Screen understanding | Claude Haiku Vision API | Fast + cheap, ~$0.01/day at 5min intervals |
| App/browser tracking | ActivityWatch | REST API at localhost:5600, has Python client |
| Audio transcription | `faster-whisper` | Local, no internet needed |
| Speaker diarization | `WhisperX` + `pyannote` | "Who said what" — later, needs GPU |
| Storage | SQLite (WAL mode) | Single file, zero infra |
| Scheduling | `APScheduler` | With job failure logging |
| AI enrichment | Claude Haiku / Sonnet | Async, decoupled from ingest |
| Notes | Obsidian or built-in markdown | Vault = folder of .md files, watched by file watcher |
| Mobile | ActivityWatch Android + BCR (rooted) | Android only — iOS is a dead end |

---

## AI / Vision Pricing (as of March 2026)

For screenshot classification (Claude Haiku Vision — fast + cheap is the right call here):

| Provider | Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|---|
| Anthropic | Claude Haiku | ~$0.80 | ~$4.00 |
| OpenAI | GPT-4o mini | ~$0.15 | ~$0.60 |
| xAI | Grok 2 Vision | $2.00 | $10.00 |
| Groq | (no vision model at scale) | — | — |

At a screenshot every 5 minutes, cost is pennies per day on any provider. GPT-4o mini is cheapest; Claude Haiku is slightly more but better at understanding context. Either works.

---

## Deployment — Local, Always-On

Everything runs locally on your Windows PC. The service boots on startup and runs silently in the background.

```
Windows startup (Task Scheduler or systemd-equivalent)
  └── Python service (runs headlessly)
        ├── screenshot_ingestor.py    (every 5 min)
        ├── activitywatch_ingestor.py (every 1 min)
        ├── audio_ingestor.py         (watches drop folder)
        ├── enrich.py                 (every 10 min, async)
        └── Web UI at localhost:8080  (review queue, search)
```

Later: wrap in a system tray icon (`pystray`) for pause/resume and status.

**On-startup options (Windows):**
- Task Scheduler (simplest — run `python main.py` on login)
- NSSM (Non-Sucking Service Manager) — runs as a proper Windows service, survives logout
- Startup folder shortcut — fine for dev, not robust enough for always-on

---

## What Doesn't Exist Yet (Custom Build Required)

| Feature | Effort |
|---|---|
| Screenshot daemon + Claude Vision pipeline | ~40 lines Python |
| ActivityWatch poller | ~30 lines Python |
| Audio folder watcher + Whisper | ~60 lines Python |
| Enrich/summarize pass | ~50 lines Python |
| Transcript review queue UI | ~100 lines (simple HTML) |
| Obsidian vault file watcher | ~40 lines Python |
| Obsidian daily note writer | ~50 lines Python |

Total custom code for v1: ~400 lines.

---

## What Already Exists (Use As-Is)

- **Screenpipe** — full screen + audio capture, OCR, Whisper transcription, REST API. Consider as an alternative/complement to rolling our own screenshot daemon.
- **ActivityWatch** — mature, just install it and poll the API.
- **faster-whisper** — pip install, works locally.
- **Khoj** — self-hosted semantic search over Obsidian vault (add later).
- **Meetily** — local meeting transcription with Markdown output (add later).

---

## Obsidian Integration

Your app writes `.md` files into the vault. Obsidian hot-reloads them instantly.

```
vault/
  daily-notes/2026-03-15.md   ← generated each morning by enrich.py
  contacts/John-Smith.md      ← auto-updated after each interaction
  weekly/2026-W11.md          ← weekly digest
```

Obsidian + Dataview plugin then becomes your read/query UI for free — no custom UI needed for most views.

---

## Honest Hard Parts

1. **Android call recording** — requires root (BCR app). Without root, calls are very hard to capture. Fallback: voice recorder app + manual drop to folder.
2. **iOS** — essentially impossible for background recording. Workaround: screen mirror to Mac via Continuity.
3. **Person identification from audio** — automated speaker → person mapping needs `pyannote` embeddings. V1: manual tagging in review queue.
4. **Data volume** — continuous screen capture generates a lot. Cap raw blobs at ~10KB, log oversized data separately, define retention policy before it fills a drive.

---

## Build Order

```
1. db.py                  — schema, WAL mode, single writer helper
2. screenshot_ingestor.py — mss capture → Claude Haiku → events table
3. activitywatch_ingestor.py — poll AW API → events table
4. audio_ingestor.py      — watchdog on drop folder → faster-whisper → transcripts
5. enrich.py              — async Claude summarization pass
6. review_queue.html      — list transcripts, assign to person, mark done
7. scheduler.py / main.py — boots everything, APScheduler with error logging
```
