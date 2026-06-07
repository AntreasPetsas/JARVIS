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
# Used only inside handle() weather branch to extract city from the query.
WEATHER_PARSE_SYSTEM = (
    "You are a weather intent parser. The user asked about weather or temperature. "
    "Call get_weather with the correct arguments. Rules:\n"
    "- Extract the city name exactly as the user stated it. "
    "Example: 'temperature in Nicosia' → city='Nicosia'.\n"
    "- If no city is mentioned, omit the city field entirely.\n"
    "Always call the tool. Never reply in plain text."
)
# Used only inside _spotify() to translate natural language → one structured tool
# call. Kept narrow (one tool, no history) and example-driven so even a small local
# model maps intent reliably. Every action the tool supports is covered here — a gap
# makes a 3B model guess (e.g. it once mapped 'play on laptop' to pause).
SPOTIFY_PARSE_SYSTEM = (
    "You translate a music command into ONE control_spotify tool call. "
    "Always call the tool; never reply with plain text.\n\n"
    "Pick the action:\n"
    "- play → resume/start when NO song, artist, or place is named "
    "('play', 'play music', 'start music', 'resume', 'unpause').\n"
    "- pause → 'pause', 'stop'.\n"
    "- next → 'next', 'skip'.\n"
    "- previous → 'previous', 'go back', 'last track', 'play the previous song'.\n"
    "- search → play something specific. Set query to what they want and pick kind:\n"
    "    kind=playlist for a mood/genre/activity ('gaming music', 'lo-fi', 'workout', 'jazz').\n"
    "    kind=artist for a band or musician by name.\n"
    "    kind=album for a named album.\n"
    "    kind=track for one specific song. For saved/liked tracks use query='liked songs'.\n"
    "- transfer → move playback to a device ('play on my laptop', 'play music on the laptop', "
    "'switch to the kitchen', 'cast to phone'). Put ONLY the device name in device (e.g. 'laptop').\n"
    "- volume → set exact level 0-100, OR direction 'up'/'down' for 'louder'/'quieter'/"
    "'turn it up/down'. 'mute' = level 0.\n"
    "- shuffle → on=true/false. repeat → mode=track/context/off. seek. list_devices.\n"
    "- now_playing → 'what's playing', 'what song is this'.\n"
    "- queue / play_next → ONLY when they literally say 'queue' or 'play next'.\n\n"
    "Examples:\n"
    "'play league of legends music' → search, query='league of legends', kind=playlist\n"
    "'play the previous song' → previous\n"
    "'play liked songs' → search, query='liked songs'\n"
    "'play music on the laptop' → transfer, device='laptop'\n"
    "'turn it up a bit' → volume, direction='up'\n"
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
    r"louder|quieter|mute|tune|shuffle|repeat|loop|queue|devices?|"
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
            or re.search(r"\b(play|pause|resume|stop|skip|track|song|playlist|volume|louder|"
                         r"quieter|mute|unmute|shuffle|repeat|loop|queue|devices?)\b", low)
            or re.search(r"what'?s playing|now playing", low)):
        await _spotify(cfg, t, low, send, history)
        return

    # --- Weather ---
    if re.search(r"\b(weather|temperature|forecast|rain|cold|hot|sunny)\b", low):
        await send({"type": "state", "state": "thinking"})
        llm = get_llm(cfg)
        if cfg.get("llm.tools", True) and getattr(llm, "supports_tools", False):
            if await _weather_llm(cfg, t, send, history, llm):
                return
        # Fallback: LLM unavailable or returned nothing — use configured home city.
        wx = await weather.get_weather(cfg.get("location.city", "London"),
                                       cfg.get("location.units", "metric"))
        await send({"type": "panel", "panel": "weather", "data": wx})
        if wx.get("ok"):
            await say(cfg, send, history, f"It's {wx['temp']}{wx['unit']} and {wx['condition'].lower()} in "
                             f"{wx['city']}, feels like {wx['feels_like']}{wx['unit']}.")
        else:
            await say(cfg, send, history, wx.get("error", "I couldn't reach the weather service."))
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


async def _weather_llm(cfg: Config, t: str, send: Send, history: History, llm) -> bool:
    """Parse a weather query via LLM to extract the city. Returns True if handled."""
    weather_tool = next((tool for tool in TOOLS if tool["name"] == "get_weather"), None)
    if not weather_tool:
        return False
    try:
        result = await llm.chat(
            [{"role": "system", "content": WEATHER_PARSE_SYSTEM},
             {"role": "user", "content": t}],
            tools=[weather_tool],
            max_tokens=100,
        )
    except Exception:  # noqa: BLE001
        return False
    calls = result.get("tool_calls") or []
    if not calls:
        return False
    args = calls[0].get("arguments") or {}
    out = await _run_tool(cfg, send, "get_weather", args)
    await say(cfg, send, history, out)
    return True


async def _spotify_llm(cfg: Config, t: str, send: Send, history: History, llm) -> bool:
    """Parse a music command via LLM tool-calling. Returns True if handled, False to fall back."""
    spotify_tool = next((tool for tool in TOOLS if tool["name"] == "control_spotify"), None)
    if not spotify_tool:
        return False
    try:
        result = await llm.chat(
            [{"role": "system", "content": SPOTIFY_PARSE_SYSTEM},
             {"role": "user", "content": t}],
            tools=[spotify_tool],
            max_tokens=200,
        )
    except Exception:  # noqa: BLE001
        return False
    calls = result.get("tool_calls") or []
    if not calls:
        return False
    args = calls[0].get("arguments") or {}
    # Validate against the tool's own action enum. A small model sometimes invents an
    # action ('switch_device'); rather than dead-end on "Unsupported music action.", we
    # bail and let the deterministic regex chain below take the command.
    valid = spotify_tool["parameters"]["properties"]["action"].get("enum", [])
    if (args.get("action") or "").lower() not in valid:
        return False
    out = await _run_spotify_tool(cfg, send, args)
    await say(cfg, send, history, out or "Done.")
    return True


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

    # LLM intent parsing — maps natural language to a structured control_spotify call.
    # More reliable than regex for phrasing variations ("play previous song", "gaming music", etc.).
    # Falls through to the regex chain below when the LLM is unavailable or returns nothing.
    llm = get_llm(cfg)
    if cfg.get("llm.tools", True) and getattr(llm, "supports_tools", False):
        await send({"type": "state", "state": "thinking"})
        if await _spotify_llm(cfg, t, send, history, llm):
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

    # List available devices.
    if re.search(r"\b(devices?)\b", low) and re.search(r"\b(what|which|list|show|available|any)\b", low):
        if not use_api:
            await say(cfg, send, history, _link_hint(web, "Device control"))
            return
        await say(cfg, send, history, (await web.list_devices())["message"])
        return

    # Transfer playback to a named device ("play on my phone", "switch to the kitchen").
    # Only treat as a transfer when the tail actually names a known device — otherwise
    # fall through so "switch to <playlist>" / "play <song>" still work.
    if use_api:
        tm = re.search(r"\b(?:play|switch|transfer|move|cast|continue)\b\s+(?:playback\s+)?"
                       r"(?:on|to)\s+(?:my\s+|the\s+)?(.+)", low)
        if tm:
            cand = tm.group(1).strip(" .?!,'\"")
            devices = await web._devices()
            match = next((d for d in devices
                          if cand in d.get("name", "").lower()
                          or d.get("name", "").lower() in cand), None)
            if match:
                res = await web.transfer_to(match.get("name", ""))
                await _spotify_done(cfg, send, history, web, use_api, res)
                return

    # Seek / restart / skip ahead — BEFORE next/skip so "skip ahead 30s" isn't a track skip.
    sk = _seek_request(low)
    if sk is not None:
        if not use_api:
            await say(cfg, send, history, _link_hint(web, "Seeking"))
            return
        mode, ms = sk
        res = await web.seek(ms) if mode == "abs" else await web.seek_relative(ms)
        await _spotify_done(cfg, send, history, web, use_api, res)
        return

    # Queue ("queue up X", "add this to the queue", "play next") — BEFORE the play parser
    # because "play next" contains "play".
    qm = re.search(r"\b(?:add\s+(?:this\s+)?(?:to\s+(?:the\s+)?)?queue|queue\s+up|queue|"
                   r"play\s+next|up\s+next)\b\s*(.*)", t, re.IGNORECASE)
    if qm:
        if not use_api:
            await say(cfg, send, history, _link_hint(web, "The queue"))
            return
        target = qm.group(1).strip(" .?!,'\"")
        if target:
            res = await web.queue_search(target)
        else:  # no target -> queue the currently playing track
            np = await web.now_playing()
            res = await web.add_to_queue(np.get("uri", "")) if np.get("uri") else \
                {"ok": False, "message": "There's nothing playing to queue."}
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

    # Play something specific ("play X", "put on Y", "shuffle Z").
    query, kind = _parse_play_query(t)
    if query:
        if not use_api:
            await say(cfg, send, history, _link_hint(web, "Playing a specific track"))
            return
        shuffle = bool(re.search(r"\bshuffle\b", low))
        if _is_liked_songs(query):
            res = await web.play_liked(shuffle=shuffle)
        else:
            # "shuffle X" with no explicit kind implies a playlist, not a single track.
            if shuffle and kind == "track" and not re.search(r"\b(song|track)\b", low):
                kind = "playlist"
            res = await web.search_and_play(query, kind, shuffle=shuffle)
        await _spotify_done(cfg, send, history, web, use_api, res)
        return

    # Shuffle on/off (no specific target).
    if re.search(r"\bshuffle\b", low):
        if not use_api:
            await say(cfg, send, history, _link_hint(web, "Shuffle"))
            return
        on = not re.search(r"\b(off|stop|disable|no|don'?t)\b", low)
        res = await web.set_shuffle(on)
        await _spotify_done(cfg, send, history, web, use_api, res)
        return

    # Repeat / loop.
    if re.search(r"\b(repeat|loop)\b", low):
        if not use_api:
            await say(cfg, send, history, _link_hint(web, "Repeat"))
            return
        if re.search(r"\b(off|stop|disable|no|don'?t)\b", low):
            mode = "off"
        elif re.search(r"\b(this|song|track|one)\b", low):
            mode = "track"
        else:
            mode = "context"
        res = await web.set_repeat(mode)
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


def _seek_request(low: str):
    """Return ('abs', ms) | ('rel', delta_ms) | None from a seek phrase."""
    if re.search(r"\b(restart|start over|start again|from the (?:top|beginning|start)|replay|"
                 r"back to the (?:top|start|beginning))\b", low):
        return ("abs", 0)
    m = re.search(r"\b(skip|jump|fast.?forward|forward|rewind|go back|back|ahead)\b"
                  r"[^0-9]*(\d{1,3})\s*(seconds?|secs?|s|minutes?|mins?|m)\b", low)
    if m:
        n = int(m.group(2))
        unit = m.group(3)
        ms = n * (60000 if unit.startswith("m") else 1000)
        backward = bool(re.search(r"\b(rewind|go back|back)\b", low))
        return ("rel", -ms if backward else ms)
    return None


def _is_liked_songs(q: str) -> bool:
    """Spotify's Liked Songs / saved library — has no searchable playlist URI."""
    low = q.strip().strip("'\"").lower()
    return low in ("liked songs", "liked", "my liked songs", "liked tracks",
                   "saved songs", "saved tracks", "my library", "favourites", "favorites")


def _parse_play_query(text: str):
    """Pull a search target out of 'play ...' / 'put on ...' / 'change ... to ...'. Returns (query, kind)."""
    m = re.search(r"\b(?:play|put on|listen to|shuffle)\b\s+(.+)", text, re.IGNORECASE)
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
    # Control words ("shuffle off", "shuffle on") aren't play targets — let the
    # dedicated shuffle/repeat toggle branches handle them.
    if q.lower() in ("", "on", "off", "music", "my music", "some music", "something",
                     "a song", "spotify", "anything", "it", "this"):
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
        shuffle = bool(args.get("shuffle"))
        kind = (args.get("kind") or "track").lower()
        if kind not in ("track", "album", "artist", "playlist"):
            kind = "track"
        if _is_liked_songs(q):
            res = await web.play_liked(shuffle=shuffle)
        else:
            res = await web.search_and_play(q, kind, shuffle=shuffle)
        if res.get("ok"):
            np = await web.now_playing()
            if np.get("ok") and np.get("title"):
                await send({"type": "panel", "panel": "nowplaying", "data": np})
        return res.get("message", "")

    if action == "volume":
        if not use_api:
            return _link_hint(web, "Volume control")
        level = args.get("level")
        if level is not None:
            return (await web.set_volume(int(level))).get("message", "")
        direction = (args.get("direction") or "").lower()
        if direction in ("up", "down"):
            return (await web.nudge_volume(15 if direction == "up" else -15)).get("message", "")
        return "What volume level?"

    if action == "shuffle":
        if not use_api:
            return _link_hint(web, "Shuffle")
        on = args.get("on")
        return (await web.set_shuffle(True if on is None else bool(on))).get("message", "")

    if action == "repeat":
        if not use_api:
            return _link_hint(web, "Repeat")
        return (await web.set_repeat((args.get("mode") or "context").lower())).get("message", "")

    if action == "seek":
        if not use_api:
            return _link_hint(web, "Seeking")
        if args.get("position_seconds") is not None:
            res = await web.seek(int(args["position_seconds"]) * 1000)
        elif args.get("delta_seconds") is not None:
            res = await web.seek_relative(int(args["delta_seconds"]) * 1000)
        else:
            return "Where should I seek to?"
        return res.get("message", "")

    if action in ("queue", "play_next"):
        if not use_api:
            return _link_hint(web, "The queue")
        q = (args.get("query") or "").strip()
        if q:
            return (await web.queue_search(q)).get("message", "")
        np = await web.now_playing()
        if not np.get("uri"):
            return "There's nothing playing to queue."
        return (await web.add_to_queue(np["uri"])).get("message", "")

    if action == "list_devices":
        if not use_api:
            return _link_hint(web, "Device control")
        return (await web.list_devices()).get("message", "")

    if action == "transfer":
        if not use_api:
            return _link_hint(web, "Device control")
        device = (args.get("device") or "").strip()
        if not device:
            return "Which device should I switch to?"
        res = await web.transfer_to(device)
        if res.get("ok"):
            np = await web.now_playing()
            if np.get("ok") and np.get("title"):
                await send({"type": "panel", "panel": "nowplaying", "data": np})
        return res.get("message", "")

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
