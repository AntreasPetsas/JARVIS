# J.A.R.V.I.S

A local, Iron Man-styled personal assistant for Windows. Python backend + browser HUD, communicating over a WebSocket. Voice-first, but fully usable by typing.

| Layer | What it adds |
|-------|--------------|
| **1 — Core** | Glowing HUD, weather, software & hobby news, reminders, Spotify control, daily briefing |
| **2 — Voice** | "Hey Jarvis" wake word, speech-to-text, edge-tts, local LLM (Ollama) with cloud fallback |

Both layers are independently runnable — start with Layer 1 and add Layer 2 when ready.

---

## Requirements

- Python 3.10+
- Windows (media-key Spotify control and the `.ps1` activation script are Windows-specific; the rest is cross-platform)
- A microphone (Layer 2 only)

---

## Setup

### Layer 1 — Core

```powershell
git clone https://github.com/your-username/Jarvis.git
cd Jarvis

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy config.example.yaml config.yaml
```

Edit `config.yaml` — at minimum set your **city**. Everything else has working defaults.

```powershell
python run.py
```

Your browser opens to `http://127.0.0.1:8765/`. Jarvis greets you with a daily briefing.

### Layer 2 — Voice + LLM

```powershell
pip install -r requirements-voice.txt
```

Then set up a local or cloud LLM (see [LLM setup](#llm-setup) below). Restart with `python run.py` — say **"Hey Jarvis"** and speak, or click the 🎙 mic for push-to-talk.

> Click anywhere in the HUD once first to unlock browser audio (browser autoplay policy).

---

## Configuration

Copy `config.example.yaml` → `config.yaml` and edit. The file is well-commented; key fields:

| Field | Purpose |
|-------|---------|
| `location.city` | Geocoded automatically for weather — no API key needed |
| `news.software.sources` | `hackernews`, `devto`, `reddit` |
| `news.hobbies` | List of `{ label, rss }` feeds for your own interests |
| `llm.provider` | `ollama` / `anthropic` / `openai` / `none` |
| `llm.model` | Model tag (e.g. `qwen2.5:7b-instruct` for Ollama) |
| `assistant.voice` | edge-tts voice ID — `en-GB-RyanNeural` sounds most Jarvis-like |
| `voice.enabled` | `false` to run text-only (no mic required) |

**Secrets** — cloud LLM API keys go in `.env` (copy `.env.example`):

```
ANTHROPIC_API_KEY=...   # if llm.provider: anthropic
OPENAI_API_KEY=...      # if llm.provider: openai
```

---

## LLM Setup

### Ollama (local, recommended)

1. Install from <https://ollama.com/download>
2. Pull a model:
   ```powershell
   ollama pull qwen2.5:7b-instruct   # good balance of quality and speed
   # or
   ollama pull llama3.2:3b           # faster, lighter
   ```
3. In `config.yaml`:
   ```yaml
   llm:
     provider: ollama
     model: qwen2.5:7b-instruct
   ```

**Model guidance by VRAM:**
- **≥8 GB VRAM** — `qwen2.5:14b`, `llama3.1:8b` fully on GPU
- **4 GB VRAM** — `qwen2.5:7b-instruct` (hybrid GPU/CPU) or `llama3.2:3b` (fits in VRAM)
- **CPU only** — `llama3.2:3b` or `qwen2.5:3b` (Q4)

### Cloud (Anthropic / OpenAI)

Set `llm.provider: anthropic` or `openai` in `config.yaml`, then add the matching key in `.env`. No extra packages needed — Jarvis uses plain HTTP.

### LLM Tool-calling

With `llm.tools: true` (default), the model can pick skills itself for phrasings the keyword rules miss — e.g. *"should I take a jacket?"* checks weather; *"any gaming news?"* pulls your hobby feed. Requires a tool-capable model (llama3.1+, qwen2.5, gpt-4o, claude-*). Disable with `llm.tools: false` for plain chat fallback.

---

## Spotify Web API (optional)

Out of the box, Spotify uses Windows media keys (play/pause/next/prev). Link the Web API to unlock **play by name, volume control, and a live now-playing panel**.

1. Create an app at <https://developer.spotify.com/dashboard>
2. Add this redirect URI in the app's settings (match your `server.port` if changed):
   ```
   http://127.0.0.1:8765/spotify/callback
   ```
3. Add to `.env`:
   ```
   SPOTIFY_CLIENT_ID=...
   SPOTIFY_CLIENT_SECRET=...
   ```
4. Restart Jarvis, then type or say **"connect Spotify"** and approve in the browser. The token is cached in `jarvis/data/` — you only do this once.

> Playback requires an active Spotify Connect device (open Spotify on any device). Track/volume control requires **Spotify Premium** (Spotify API restriction).

---

## Voice Tuning

All in `config.yaml`:

| Setting | Effect |
|---------|--------|
| `assistant.wake_sensitivity` | Raise if "Hey Jarvis" won't trigger; lower if it false-fires |
| `voice.silence_rms` | Raise if Jarvis cuts you off; lower if it waits too long |
| `voice.stt_model` | `base` → `small` for better accuracy (slower) |
| `voice.stt_device` | `cuda` if you want STT on GPU (frees CPU, uses VRAM) |

---

## Try these commands

```
brief me                          → full daily briefing
what's the weather in Tokyo
any gaming news / tech headlines
remind me to push the branch
reminders
open spotify / pause / next
play bohemian rhapsody            → Web API required
volume 40 / louder / quieter      → Web API required
what's playing                    → Web API required
```

Anything else is answered by the LLM (Layer 2).

---

## Architecture

```
web/ (browser HUD)  ──WebSocket──  jarvis/server.py
  app.js reactor + panels                 │
                                    router.py  ── keyword rules → skill
                                    ├─ skills/weather.py    (Open-Meteo, no key)
                                    ├─ skills/news.py       (HN / DEV.to / Reddit / RSS)
                                    ├─ skills/spotify.py    (media keys)
                                    ├─ skills/spotify_api.py (Web API, OAuth)
                                    ├─ skills/reminders.py  (JSON store)
                                    ├─ skills/briefing.py   (orchestrates + LLM)
                                    └─ llm/                 (ollama|anthropic|openai|none)
```

Voice (Layer 2) and typed text both enter `router.handle()` — they share one brain and one conversation history.
