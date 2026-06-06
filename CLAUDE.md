# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running Jarvis

```powershell
# Activate the venv first
.\.venv\Scripts\Activate.ps1

# Start the server (opens browser automatically)
python run.py
```

The HUD opens at `http://127.0.0.1:8765/`. Config is read from `config.yaml` (falls back to `config.example.yaml`).

## Setup

```powershell
# Layer 1 (core, always required)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy config.example.yaml config.yaml   # then edit

# Layer 2 (voice pipeline, optional)
pip install -r requirements-voice.txt
```

Secrets go in `.env` (copy `.env.example`). Only needed if `llm.provider` is `anthropic` or `openai`.

## Architecture

```
web/           Browser HUD (HTML + CSS + app.js) — Iron Man-styled UI
jarvis/
  run.py               Entry point: loads config, starts uvicorn, opens browser
  config.py            Config loader — deep-merges config.yaml over _DEFAULTS; dotted cfg.get('a.b')
  server.py            FastAPI app: serves web/ static files + /ws WebSocket endpoint
                       Hub class broadcasts to all connected HUD clients
                       Also hosts /spotify/login + /spotify/callback (Web API OAuth)
  router.py            Intent router: keyword regex → skill; falls back to LLM for chat
  skills/
    weather.py         Open-Meteo (geocode city → lat/lon → forecast, no API key)
    news.py            HackerNews / DEV.to / Reddit / RSS feeds
    spotify.py         Media key simulation (fallback — opens via spotify: protocol)
    spotify_api.py     Spotify Web API (OAuth): search&play, volume, now-playing; token in data/
    reminders.py       JSON-file reminder store
    briefing.py        Orchestrates all skills + LLM into a spoken daily briefing
  llm/
    providers.py       OllamaProvider / AnthropicProvider / OpenAIProvider / NullProvider
                       All use plain httpx — no vendor SDKs installed
                       complete() = plain text; chat(messages, tools) = tool-calling
    tools.py           Provider-neutral tool schemas the LLM can call (dispatched in router)
  voice/
    assistant.py       VoiceAssistant: background thread, mic → wake word → STT → router
    wake.py            openWakeWord wrapper ("hey_jarvis" model)
    stt.py             faster-whisper Transcriber
    tts.py             edge-tts → MP3 bytes sent to browser as base64
```

## Key data flows

**Typed command:** `WebSocket /ws` → `server.py` receives `{type:"command"}` → `router.handle()` → skill or LLM → `send()` pushes panel/say/transcript messages back to HUD.

**Voice command:** `VoiceAssistant` thread → wake word fires → microphone records → `Transcriber.transcribe()` → `router.handle()` (same path as typed).

**TTS:** `router.say()` calls `tts.synthesize()` → base64 MP3 embedded in the `say` WebSocket message → browser plays it and then signals playback done to re-idle the HUD.

## Adding a new skill

1. Create `jarvis/skills/myskill.py` with an `async` function.
2. Add a keyword branch in `router.py` `handle()` — match on `low` (lowercased text), call `await send({"type": "panel", ...})` and `await say(cfg, send, history, text)`. (`say()` records the spoken reply into the rolling `History`; `handle()` already records the user turn.)
3. (Optional) Expose it to the tool-calling LLM: add a schema to `llm/tools.py` and a branch in `router._run_tool()` so the model can pick it for off-keyword phrasings.

## LLM tool-calling (the model picks skills)

Keyword rules in `handle()` are the fast path. When none match, the fallback decides between two single-call paths using `TOOL_HINT_RE` (a cheap regex of tool-domain words like "rain", "play", "remind"): if the text hints at a skill the keyword rules missed, it runs `router._agent()` — `llm.chat(messages, tools=TOOLS)`; if the model returns `tool_calls`, `_run_tool()` executes the matching skill (pushing the same panels) and the model narrates via a second `complete()`. Otherwise the request is plain conversation: `llm.chat(messages)` with no tools. The gate exists because small local models grab a tool whenever tools are attached, so casual chat ("say hi to my mum") must not even see them. Gated by `llm.tools` + the provider's `supports_tools`.

Both LLM paths receive the rolling `History` (recent user/assistant turns, capped by `llm.history_turns`), so Jarvis holds multi-turn conversations. The user turn is recorded in `handle()`; every spoken reply is recorded in `say()`. One shared `History` lives in `server.create_app()`, so typed and voice input share one memory.

## Config system

`Config.get(path, default)` does dotted-path lookups (e.g. `cfg.get("llm.model")`). Settings in `config.yaml` deep-merge over `_DEFAULTS` in `config.py`. There is no hot-reload; restart the server after config changes.

## Hardware context (this machine)

RTX 3050 Ti (4 GB VRAM), 32 GB RAM. Recommended Ollama models: `qwen2.5:7b-instruct` (quality, partial GPU offload) or `llama3.2:3b` (speed, fits in VRAM). Whisper STT: `base` or `small`.
