"""FastAPI server: serves the HUD and bridges it to Jarvis over a WebSocket."""
from __future__ import annotations

import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .config import Config, load_config
from .memory_store import JsonlStore
from .router import History, handle, say
from .skills import reminders
from .skills.briefing import spoken_briefing
from .skills.onboarding import InterviewState
from .skills.spotify_api import get_spotify

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
DATA_DIR = Path(__file__).resolve().parent / "data"


def _spotify_page(title: str, body: str) -> str:
    """A small HUD-themed page shown after the Spotify OAuth redirect."""
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Jarvis · Spotify</title><style>
  body {{ margin:0; height:100vh; display:flex; align-items:center; justify-content:center;
         background: radial-gradient(900px 600px at 50% -10%, #0a2236, #061320 45%, #02060c);
         color:#cfeefc; font-family:"Segoe UI",sans-serif; }}
  .card {{ text-align:center; padding:40px 48px; border:1px solid rgba(54,230,255,0.35);
          border-radius:12px; background:rgba(10,28,44,0.5); box-shadow:0 0 40px rgba(54,230,255,0.15); }}
  h1 {{ color:#36e6ff; font-weight:700; letter-spacing:2px; margin:0 0 12px; }}
  p {{ color:#7fb4c8; margin:0; }}
</style></head><body><div class="card"><h1>{title}</h1><p>{body}</p></div></body></html>"""


class Hub:
    """Tracks connected HUD clients so skills (and the voice loop) can broadcast."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def broadcast(self, msg: dict) -> None:
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(json.dumps(msg))
            except (WebSocketDisconnect, RuntimeError):
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    hub = Hub()
    # One shared conversation memory so typed and voice input share a single brain.
    # Persist it to disk (data/history.jsonl) so it survives restarts, unless disabled.
    store = JsonlStore(DATA_DIR / "history.jsonl") if cfg.get("llm.persist_history", True) else None
    history = History(cfg.get("llm.history_turns", 6), store=store)
    # Shared "get to know me" interview progress (typed + voice drive the same one).
    interview = InterviewState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Start the microphone loop (wake word + STT) if enabled and deps present.
        app.state.voice = None
        if cfg.get("voice.enabled", True):
            try:
                from .voice.assistant import VoiceAssistant

                loop = asyncio.get_running_loop()

                async def on_command(text: str) -> None:
                    await handle(cfg, text, hub.broadcast, history, interview)

                va = VoiceAssistant(cfg, loop, hub.broadcast, on_command)
                va.start()
                app.state.voice = va
                print("[voice] microphone loop started — say 'Hey Jarvis'")
            except Exception as e:  # noqa: BLE001 — voice is optional
                print(f"[voice] unavailable ({e}); running in text-only mode")
        yield
        if app.state.voice is not None:
            app.state.voice.stop()

    app = FastAPI(title="Jarvis", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.hub = hub
    app.state.spotify_state = None

    @app.get("/spotify/login")
    async def spotify_login():
        web = get_spotify(cfg)
        if not web.configured():
            return HTMLResponse(_spotify_page(
                "Spotify isn't configured",
                "Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to your .env file "
                "(then restart Jarvis) and reload this page."), status_code=400)
        state = secrets.token_urlsafe(16)
        app.state.spotify_state = state
        return RedirectResponse(web.auth_url(state))

    @app.get("/spotify/callback")
    async def spotify_callback(code: str = "", state: str = "", error: str = ""):
        if error:
            return HTMLResponse(_spotify_page("Authorization cancelled", error), status_code=400)
        if not code or not state or state != app.state.spotify_state:
            return HTMLResponse(_spotify_page(
                "Authorization failed",
                "State mismatch or missing code. Start again from /spotify/login."),
                status_code=400)
        app.state.spotify_state = None
        ok = await get_spotify(cfg).exchange_code(code)
        if ok:
            await hub.broadcast({"type": "voice_status", "ok": True,
                                 "message": "Spotify linked — try 'play <song>' or 'what's playing'."})
            return HTMLResponse(_spotify_page(
                "Spotify linked ✓", "You can close this tab and talk to Jarvis."))
        return HTMLResponse(_spotify_page(
            "Authorization failed",
            "Token exchange failed — check your client secret and that the redirect URI "
            "matches your Spotify app exactly."), status_code=400)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        hub.clients.add(websocket)

        async def send(msg: dict) -> None:
            await websocket.send_text(json.dumps(msg))

        await send({"type": "hello", "name": cfg.get("assistant.name", "Jarvis")})
        await send({"type": "state", "state": "idle"})
        await send({"type": "voice_status", "ok": app.state.voice is not None,
                    "message": ("Voice online — say 'Hey Jarvis', or click the mic."
                                if app.state.voice is not None else "Voice off — type below.")})

        briefing_task: asyncio.Task | None = None
        if cfg.get("briefing.on_startup", True):
            briefing_task = asyncio.create_task(_run_briefing(cfg, send, history))

        async def cancel_briefing() -> None:
            nonlocal briefing_task
            if briefing_task is not None and not briefing_task.done():
                briefing_task.cancel()
                await send({"type": "stop_audio"})
            briefing_task = None

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                mtype = msg.get("type")
                if mtype == "command":
                    await cancel_briefing()
                    asyncio.create_task(handle(cfg, msg.get("text", ""), send, history, interview))
                elif mtype == "briefing":
                    await cancel_briefing()
                    briefing_task = asyncio.create_task(_run_briefing(cfg, send, history))
                elif mtype == "listen":
                    if app.state.voice is not None:
                        app.state.voice.trigger_listen()
                elif mtype == "speaking":
                    if app.state.voice is not None:
                        if msg.get("on"):
                            app.state.voice.mute(msg.get("seconds"))
                        else:
                            app.state.voice.unmute()
                elif mtype == "reminder_done":
                    reminders.complete_reminder(str(msg.get("id", "")))
                    await send({"type": "panel", "panel": "reminders",
                                "data": reminders.list_reminders()})
                elif mtype == "ping":
                    await send({"type": "pong"})
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(websocket)

    # Mounted last so the /ws route keeps priority over the catch-all static handler.
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app


async def _run_briefing(cfg: Config, send, history: History | None = None) -> None:
    try:
        await send({"type": "state", "state": "thinking"})
        result = await spoken_briefing(cfg)
        data = result["data"]
        await send({"type": "panel", "panel": "weather", "data": data.get("weather", {})})
        await send({"type": "panel", "panel": "software_news", "data": data.get("software_news", [])})
        await send({"type": "panel", "panel": "hobby_news", "data": data.get("hobby_news", [])})
        await send({"type": "panel", "panel": "reminders", "data": data.get("reminders", [])})
        await say(cfg, send, history, result["text"])  # edge-tts; HUD idles when playback ends
    except (WebSocketDisconnect, RuntimeError):
        pass
