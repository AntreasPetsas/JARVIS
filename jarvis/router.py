"""Intent routing — map a text command (typed or transcribed) to a skill.

Layer 1 uses fast keyword rules for the action skills and falls back to the LLM
for open conversation. Voice transcripts (Layer 2) enter through the same
`handle()` function, so speech and text share one brain.
"""
from __future__ import annotations

import re
import webbrowser
from collections import deque
from typing import Awaitable, Callable

from .config import Config
from .llm import TOOLS, get_llm
from .skills import news, reminders, spotify, weather
from .skills.briefing import spoken_briefing
from .skills.spotify_api import get_spotify

Send = Callable[[dict], Awaitable[None]]


class History:
    """A small rolling memory of the conversation, shared by typed and voice input.

    Holds the last `turns` exchanges as provider-neutral {role, content} messages,
    so it can be handed straight to `LLMProvider.chat()`. Capped so a long session
    never blows the context window of a small local model.
    """

    def __init__(self, turns: int = 6):
        self._msgs: deque[dict] = deque(maxlen=max(1, turns) * 2)

    def add(self, role: str, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._msgs.append({"role": role, "content": text})

    def messages(self) -> list[dict]:
        return list(self._msgs)

CHAT_SYSTEM = (
    "You are Jarvis, a concise, witty British AI butler for a software engineer. "
    "Keep replies short and conversational — they will be spoken aloud. No markdown."
)
AGENT_SYSTEM = (
    "You are Jarvis, a witty British AI butler for a software engineer. You have tools "
    "to check the weather, fetch news, manage reminders, control Spotify, and give a "
    "daily briefing. Only call a tool when the user explicitly asks for one of those "
    "specific capabilities. For greetings, messages to pass on, general questions, or "
    "anything conversational, reply directly without calling any tool. "
    "Replies are spoken aloud, so keep them short with no markdown."
)
AGENT_NARRATE_SYSTEM = (
    "You are Jarvis, a concise, witty British AI butler. Turn the tool results into a "
    "short spoken reply of one or two sentences. No markdown, no lists."
)

# "add" phrasings — matched against the original text so casing is preserved.
ADD_RE = re.compile(
    r"(?:remind me(?:\s+to)?|remember(?:\s+to)?|add(?:\s+a)?\s+(?:reminder|task|to-?do)"
    r"(?:\s+(?:to|for|about))?|new\s+(?:reminder|task)|note(?:\s+that)?|i\s+(?:need|have)\s+to)\s+(.+)",
    re.IGNORECASE,
)
# "complete" phrasings — kept explicit (e.g. "check off", not bare "check") to avoid false hits.
COMPLETE_RE = re.compile(
    r"(?:mark|complete|completed|finish(?:ed)?|tick\s+off|check\s+off|cross\s+off|done\s+with)"
    r"\s+(?:the\s+|my\s+)?(?:reminder\s+|task\s+)?(.+)",
    re.IGNORECASE,
)
# Soft hints that a fallback request *might* want a skill the keyword rules above
# missed (e.g. "is it going to pour today?"). Small local models grab a tool whenever
# tools are attached, so we only expose tools to the LLM when one of these appears —
# otherwise the request goes straight to plain conversation. One LLM call either way.
TOOL_HINT_RE = re.compile(
    r"\b(weather|temperature|forecast|rain|snow|wind|storm|umbrella|jacket|coat|sunny|"
    r"cloud|degrees|pour|drizzle|chilly|freezing|outside|"
    r"news|headlines?|stor(?:y|ies)|article|happening|"
    r"play|pause|resume|skip|song|track|album|artist|playlist|music|spotify|volume|"
    r"louder|quieter|mute|tune|"
    r"remind|reminders?|tasks?|to-?dos?|remember|forget|"
    r"brief|briefing|catch me up|update me)\b",
    re.IGNORECASE,
)


async def handle(cfg: Config, text: str, send: Send, history: History | None = None) -> None:
    t = text.strip()
    if not t:
        return
    history = history if history is not None else History()
    low = t.lower()
    await send({"type": "transcript", "role": "user", "text": t})
    history.add("user", t)

    # --- Daily briefing ---
    if re.search(r"\b(brief|briefing|catch me up|what'?s going on|update me)\b", low):
        await send({"type": "state", "state": "thinking"})
        result = await spoken_briefing(cfg)
        await _push_panels(result["data"], send)
        await say(cfg, send, history,result["text"])
        return

    # --- Spotify / music ---
    if ("spotify" in low or "music" in low
            or re.search(r"\b(play|pause|resume|stop|skip|track|song|playlist|volume|louder|quieter|mute|unmute)\b", low)
            or re.search(r"what'?s playing|now playing", low)):
        await _spotify(cfg, t, low, send, history)
        return

    # --- Weather ---
    if re.search(r"\b(weather|temperature|forecast|rain|cold|hot|sunny)\b", low):
        await send({"type": "state", "state": "thinking"})
        wx = await weather.get_weather(cfg.get("location.city", "London"),
                                       cfg.get("location.units", "metric"))
        await send({"type": "panel", "panel": "weather", "data": wx})
        if wx.get("ok"):
            await say(cfg, send, history,f"It's {wx['temp']}{wx['unit']} and {wx['condition'].lower()} in "
                             f"{wx['city']}, feels like {wx['feels_like']}{wx['unit']}.")
        else:
            await say(cfg, send, history,wx.get("error", "I couldn't reach the weather service."))
        return

    # --- Hobby news (checked before generic "news" so it isn't swallowed) ---
    if re.search(r"\bhobb(?:y|ies)\b", low):
        await send({"type": "state", "state": "thinking"})
        hobbies = cfg.get("news.hobbies", [])
        hb = await news.get_hobby_news(hobbies)
        await send({"type": "panel", "panel": "hobby_news", "data": hb})
        if not hobbies:
            await say(cfg, send, history,"You haven't set up any hobby feeds yet — add them under "
                             "news.hobbies in config.yaml.")
        elif re.search(r"\bhow many\b", low):
            await say(cfg, send, history,f"You have {len(hb)} hobby stories across {len(hobbies)} "
                             f"feed{'s' if len(hobbies) != 1 else ''}.")
        elif hb:
            await say(cfg, send, history,"From your hobbies: " + "; ".join(a["title"] for a in hb[:3]) + ".")
        else:
            await say(cfg, send, history,"I couldn't pull any hobby stories just now.")
        return

    # --- Software news / headlines ---
    if re.search(r"\b(news|headlines|stories|happening)\b", low):
        await send({"type": "state", "state": "thinking"})
        sw = await news.get_software_news(cfg.get("news.software", {}))
        await send({"type": "panel", "panel": "software_news", "data": sw})
        if re.search(r"\bhow many\b", low):
            await say(cfg, send, history,f"I've got {len(sw)} software stories for you.")
        elif sw:
            await say(cfg, send, history,"Top in software right now: " + "; ".join(a["title"] for a in sw[:3]) + ".")
        else:
            await say(cfg, send, history,"I couldn't pull any headlines just now.")
        return

    # --- Reminders / to-do ---
    if (re.search(r"\b(remind|reminders?|remember|to-?dos?|tasks?)\b", low)
            or ADD_RE.search(t) or COMPLETE_RE.search(low)):
        # Add first, so "remind me to complete X" is an add, not a completion.
        am = ADD_RE.search(t)
        if am and am.group(1).strip():
            item = reminders.add_reminder(am.group(1).strip(" ,."))
            await send({"type": "panel", "panel": "reminders", "data": reminders.list_reminders()})
            await say(cfg, send, history,f"Noted: {item['text']}.")
            return
        # Complete / mark done
        cm = COMPLETE_RE.search(low)
        if cm:
            target = re.sub(r"\s*(?:as\s+)?(?:done|complete|completed|finished|off)\s*$", "",
                            cm.group(1), flags=re.IGNORECASE).strip(" ,.")
            item = reminders.complete_reminder(target) if target else None
            await send({"type": "panel", "panel": "reminders", "data": reminders.list_reminders()})
            await say(cfg, send, history,f"Marked done: {item['text']}." if item
                             else (f"I couldn't find a reminder matching '{target}'." if target
                                   else "Which reminder should I mark done?"))
            return
        # Otherwise list
        items = reminders.list_reminders()
        await send({"type": "panel", "panel": "reminders", "data": items})
        await say(cfg, send, history,("Your reminders: " + "; ".join(r["text"] for r in items[:5]) + ".")
                         if items else "You have no reminders. Try: 'remind me to push the branch'.")
        return

    # --- Fallback: let the model answer or pick a skill itself (tool-calling) ---
    await send({"type": "state", "state": "thinking"})
    llm = get_llm(cfg)
    if not await llm.available():
        # e.g. "Ollama is running, but 'qwen2.5:7b-instruct' isn't pulled. Run ollama pull ..."
        await say(cfg, send, history, await llm.hint())
        return
    # Only hand the model tools when the request plausibly wants one; otherwise a
    # small model would call a skill for plain chitchat. Either branch is one call.
    if (cfg.get("llm.tools", True) and getattr(llm, "supports_tools", False)
            and TOOL_HINT_RE.search(low)):
        await _agent(cfg, t, send, history)
        return
    try:
        result = await llm.chat([{"role": "system", "content": CHAT_SYSTEM}, *history.messages()],
                                max_tokens=cfg.get("llm.max_tokens", 500))
        reply = result.get("text", "").strip()
    except Exception:  # noqa: BLE001
        reply = "I reached the model but the request failed — check the Ollama window for errors."
    await say(cfg, send, history, reply or "I'm not sure how to help with that one, sir.")


async def _spotify(cfg: Config, t: str, low: str, send: Send, history: History) -> None:
    """Music control. Uses the Spotify Web API when linked; else Windows media keys."""
    web = get_spotify(cfg)
    use_api = web.configured() and web.authorized()

    # Link / authorize the Web API.
    if "spotify" in low and re.search(r"\b(connect|link|authori[sz]e|sign in|log in|setup|set up)\b", low):
        if not web.configured():
            await say(cfg, send, history, "Spotify Web control isn't set up yet. Add your Spotify app's "
                                 "client ID and secret to the .env file, then ask me to connect Spotify.")
            return
        host, port = cfg.get("server.host", "127.0.0.1"), cfg.get("server.port", 8765)
        try:
            webbrowser.open(f"http://{host}:{port}/spotify/login")
        except Exception:  # noqa: BLE001
            pass
        await say(cfg, send, history, "Opening the Spotify sign-in page in your browser. "
                             "Approve access and you're all set.")
        return

    # Open the desktop app.
    if "open" in low and "spotify" in low:
        res = spotify.open_spotify(cfg.get("spotify.exe_path", ""))
        await say(cfg, send, history, res["message"])
        return

    # What's playing?
    if re.search(r"what'?s playing|now playing|what (?:song|track)|which (?:song|track)|"
                 r"current (?:song|track)|what am i listening", low):
        if not use_api:
            await say(cfg, send, history, _link_hint(web))
            return
        np = await web.now_playing()
        await send({"type": "panel", "panel": "nowplaying", "data": np})
        await say(cfg, send, history, np.get("message", "I couldn't tell what's playing."))
        return

    # Volume.
    vol = _volume_request(low)
    if vol is not None:
        if not use_api:
            await say(cfg, send, history, _link_hint(web, "Volume control"))
            return
        kind, n = vol
        res = await web.set_volume(n) if kind == "set" else await web.nudge_volume(n)
        await say(cfg, send, history, res["message"])
        return

    # Skip / previous.
    if re.search(r"\b(next|skip)\b", low):
        res = await web.next() if use_api else spotify.next_track()
        await _spotify_done(cfg, send, history, web, use_api, res)
        return
    if re.search(r"\b(previous|prev|back|last)\b", low):
        res = await web.previous() if use_api else spotify.prev_track()
        await _spotify_done(cfg, send, history, web, use_api, res)
        return

    # Play something specific ("play X", "put on Y", "listen to Z").
    query, kind = _parse_play_query(t)
    if query:
        if not use_api:
            await say(cfg, send, history, _link_hint(web, "Playing a specific track"))
            return
        res = await web.play_liked() if _is_liked_songs(query) else await web.search_and_play(query, kind)
        await _spotify_done(cfg, send, history, web, use_api, res)
        return

    # Pause / stop.
    if re.search(r"\b(pause|stop)\b", low):
        res = await web.pause() if use_api else spotify.play_pause()
        await say(cfg, send, history, res["message"])
        return

    # Bare play / resume -> toggle.
    res = await web.toggle() if use_api else spotify.play_pause()
    await say(cfg, send, history, res["message"])


async def _spotify_done(cfg: Config, send: Send, history: History, web, use_api: bool, res: dict) -> None:
    """Speak the result, then refresh the Now-Playing panel if the API is live."""
    await say(cfg, send, history, res.get("message", "Done."))
    if use_api and res.get("ok"):
        np = await web.now_playing()
        if np.get("ok") and np.get("title"):
            await send({"type": "panel", "panel": "nowplaying", "data": np})


def _link_hint(web, what: str = "Spotify Web control") -> str:
    if not web.configured():
        return (f"{what} needs setup — add your Spotify app's client ID and secret to the "
                ".env file, then say 'connect Spotify'.")
    return f"{what} needs linking. Say 'connect Spotify' and approve access in the browser."


def _volume_request(low: str):
    """Return ('set', pct) | ('nudge', delta) | None from a volume phrase."""
    if re.search(r"\bunmute\b", low):
        return ("set", 40)
    if re.search(r"\bmute\b", low):
        return ("set", 0)
    m = re.search(r"\b(?:set\s+(?:the\s+)?)?(?:volume|vol)\b\s*(?:to|=|at|level)?\s*(\d{1,3})", low)
    if m:
        return ("set", int(m.group(1)))
    if re.search(r"\b(louder|turn it up|turn up|volume up|crank|pump it up|raise)\b", low):
        return ("nudge", 15)
    if re.search(r"\b(quieter|softer|turn it down|turn down|volume down|lower)\b", low):
        return ("nudge", -15)
    return None


def _is_liked_songs(q: str) -> bool:
    """Spotify's Liked Songs / saved library — has no searchable playlist URI."""
    low = q.strip().strip("'\"").lower()
    return low in ("liked songs", "liked", "my liked songs", "liked tracks",
                   "saved songs", "saved tracks", "my library", "favourites", "favorites")


def _parse_play_query(text: str):
    """Pull a search target out of 'play ...' / 'put on ...' / 'change ... to ...'. Returns (query, kind)."""
    m = re.search(r"\b(?:play|put on|listen to)\b\s+(.+)", text, re.IGNORECASE)
    if not m:
        # "change/switch [the] [playlist/album/artist] to X"
        cm = re.search(r"\b(?:change|switch)\b(.{0,40}?)\bto\s+(.+)", text, re.IGNORECASE)
        if cm:
            middle = cm.group(1).lower()
            q = cm.group(2).strip(" .?!,'\"")
            kind = "track"
            for k in ("playlist", "album", "artist"):
                if k in middle:
                    kind = k
                    break
            if q and q.lower() not in ("music", "something", "a song", "spotify", "anything"):
                return q, kind
        return "", "track"
    q = m.group(1).strip(" .?!,")
    kind = "track"
    mk = re.match(r"(?:the\s+)?(album|artist|playlist|song|track)\s+(?:called\s+|named\s+)?(.+)",
                  q, re.IGNORECASE)
    if mk:
        kind = {"song": "track", "track": "track"}.get(mk.group(1).lower(), mk.group(1).lower())
        q = mk.group(2)
    q = re.sub(r"^(?:me\s+|some\s+|a\s+|the\s+)+", "", q, flags=re.IGNORECASE).strip(" .?!,")
    if q.lower() in ("", "music", "something", "a song", "spotify", "anything"):
        return "", kind
    return q, kind


# --- LLM agent: the model picks a skill (tool-call), we run it, it narrates -----
async def _agent(cfg: Config, t: str, send: Send, history: History) -> None:
    llm = get_llm(cfg)
    max_tokens = cfg.get("llm.max_tokens", 500)
    try:
        result = await llm.chat(
            [{"role": "system", "content": AGENT_SYSTEM}, *history.messages()],
            tools=TOOLS, max_tokens=max_tokens)
    except Exception:  # noqa: BLE001
        result = {"text": "", "tool_calls": []}

    calls = result.get("tool_calls") or []
    if not calls:
        # No tool wanted — treat it as ordinary conversation.
        reply = result.get("text", "").strip()
        if not reply:
            try:
                chat = await llm.chat([{"role": "system", "content": CHAT_SYSTEM}, *history.messages()],
                                      max_tokens=max_tokens)
                reply = chat.get("text", "").strip()
            except Exception:  # noqa: BLE001
                reply = ""
        await say(cfg, send, history, reply or "I'm not sure how to help with that one, sir.")
        return

    # Run the chosen tool(s) — capped so a confused model can't loop forever.
    results: list[str] = []
    for c in calls[:3]:
        out = await _run_tool(cfg, send, c.get("name", ""), c.get("arguments") or {})
        results.append(f"{c.get('name', '?')}: {out}")

    # Let the model narrate the results in Jarvis's voice; fall back to raw output.
    narrate = (f'The user said: "{t}". You used tools and got:\n' + "\n".join(results)
               + "\nReply in one or two natural spoken sentences.")
    try:
        reply = (await llm.complete(AGENT_NARRATE_SYSTEM, narrate, min(max_tokens, 300))).strip()
    except Exception:  # noqa: BLE001
        reply = ""
    if not reply:
        reply = " ".join(r.split(": ", 1)[-1] for r in results)
    await say(cfg, send, history, reply)


async def _run_tool(cfg: Config, send: Send, name: str, args: dict) -> str:
    """Execute a tool the model picked: run the skill, push its panel, return a summary."""
    if name == "get_weather":
        city = (args.get("city") or cfg.get("location.city", "London"))
        wx = await weather.get_weather(city, cfg.get("location.units", "metric"))
        await send({"type": "panel", "panel": "weather", "data": wx})
        if wx.get("ok"):
            return (f"{wx['city']}: {wx['condition']}, {wx['temp']}{wx['unit']} "
                    f"(feels {wx['feels_like']}{wx['unit']}), high {wx['high']} low {wx['low']}, "
                    f"{wx.get('precip_chance', 0)}% precip.")
        return wx.get("error", "weather unavailable")

    if name == "get_news":
        kind = (args.get("kind") or "software").lower()
        if kind == "hobby":
            items = await news.get_hobby_news(cfg.get("news.hobbies", []))
            await send({"type": "panel", "panel": "hobby_news", "data": items})
        else:
            items = await news.get_software_news(cfg.get("news.software", {}))
            await send({"type": "panel", "panel": "software_news", "data": items})
        return ("Top stories: " + "; ".join(a["title"] for a in items[:5])) if items else "No stories found."

    if name == "add_reminder":
        text = (args.get("text") or "").strip()
        if not text:
            return "No reminder text was given."
        item = reminders.add_reminder(text)
        await send({"type": "panel", "panel": "reminders", "data": reminders.list_reminders()})
        return f"Added reminder: {item['text']}."

    if name == "list_reminders":
        items = reminders.list_reminders()
        await send({"type": "panel", "panel": "reminders", "data": items})
        return ("Open reminders: " + "; ".join(r["text"] for r in items)) if items else "No reminders."

    if name == "complete_reminder":
        q = (args.get("query") or "").strip()
        item = reminders.complete_reminder(q) if q else None
        await send({"type": "panel", "panel": "reminders", "data": reminders.list_reminders()})
        return f"Marked done: {item['text']}." if item else f"No reminder matched '{q}'."

    if name == "control_spotify":
        return await _run_spotify_tool(cfg, send, args)

    if name == "daily_briefing":
        result = await spoken_briefing(cfg)
        await _push_panels(result["data"], send)
        return result["text"]

    return f"Unknown tool '{name}'."


async def _run_spotify_tool(cfg: Config, send: Send, args: dict) -> str:
    web = get_spotify(cfg)
    use_api = web.configured() and web.authorized()
    action = (args.get("action") or "play").lower()

    if action == "now_playing":
        if not use_api:
            return _link_hint(web)
        np = await web.now_playing()
        await send({"type": "panel", "panel": "nowplaying", "data": np})
        return np.get("message", "")

    if action == "search":
        q = (args.get("query") or "").strip()
        if not q:
            return "What should I play?"
        if not use_api:
            return _link_hint(web, "Playing a specific track")
        res = await web.play_liked() if _is_liked_songs(q) else await web.search_and_play(q)
        if res.get("ok"):
            np = await web.now_playing()
            if np.get("ok") and np.get("title"):
                await send({"type": "panel", "panel": "nowplaying", "data": np})
        return res.get("message", "")

    if action == "volume":
        if not use_api:
            return _link_hint(web, "Volume control")
        level = args.get("level")
        if level is None:
            return "What volume level?"
        return (await web.set_volume(int(level))).get("message", "")

    if action in ("next", "previous", "pause", "play"):
        if use_api:
            method = {"next": web.next, "previous": web.previous,
                      "pause": web.pause, "play": web.toggle}[action]
            res = await method()
        else:
            res = {"next": spotify.next_track, "previous": spotify.prev_track,
                   "pause": spotify.play_pause, "play": spotify.play_pause}[action]()
        return res.get("message", "Done.")

    return "Unsupported music action."


async def say(cfg: Config, send: Send, history: History | None, text: str) -> None:
    # The HUD returns itself to "idle" when speech playback finishes, so we
    # deliberately do not send a trailing idle here (it would cut speech short).
    if history is not None:
        history.add("assistant", text)
    await send({"type": "state", "state": "speaking"})
    msg = {"type": "say", "text": text}
    audio = await _tts_b64(cfg, text)
    if audio:
        msg["audio"] = audio
        msg["mime"] = "audio/mpeg"
    await send(msg)
    await send({"type": "transcript", "role": "jarvis", "text": text})


async def _tts_b64(cfg: Config, text: str) -> str | None:
    """edge-tts -> base64 MP3, or None so the browser uses its built-in voice."""
    if not cfg.get("voice.tts", True):
        return None
    try:
        from .voice import tts
    except Exception:  # voice deps not installed
        return None
    try:
        audio = await tts.synthesize(text, cfg.get("assistant.voice", "en-GB-RyanNeural"))
        import base64
        return base64.b64encode(audio).decode("ascii")
    except Exception:  # network/voice error — fall back to Web Speech
        return None


async def _push_panels(data: dict, send: Send) -> None:
    await send({"type": "panel", "panel": "weather", "data": data.get("weather", {})})
    await send({"type": "panel", "panel": "software_news", "data": data.get("software_news", [])})
    await send({"type": "panel", "panel": "hobby_news", "data": data.get("hobby_news", [])})
    await send({"type": "panel", "panel": "reminders", "data": data.get("reminders", [])})
