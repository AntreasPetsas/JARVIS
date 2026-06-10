"""Tool schemas the LLM can call to drive Jarvis's skills.

These are provider-neutral (OpenAI-style JSON-Schema `parameters`); each provider
in `providers.py` translates them to its own wire format. The dispatcher that
actually runs a chosen tool lives in `router.py` (it already owns every skill).
"""
from __future__ import annotations

TOOLS: list[dict] = [
    {
        "name": "get_weather",
        "description": "Current weather and today's forecast for a city. Use ONLY when the "
                       "user actually asks about weather, temperature, rain, what to wear, or "
                       "whether it's a good day to go outside. Never for greetings or chit-chat.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string",
                         "description": "City name. Omit to use the user's home city."},
            },
            "required": [],
        },
    },
    {
        "name": "get_news",
        "description": "Fetch the latest headlines. kind='software' for tech/programming "
                       "news; kind='hobby' for the user's configured hobby feeds.",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["software", "hobby"]},
            },
            "required": ["kind"],
        },
    },
    {
        "name": "add_reminder",
        "description": "Add a to-do / reminder item to the user's list.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The reminder text."}},
            "required": ["text"],
        },
    },
    {
        "name": "list_reminders",
        "description": "List the user's current open reminders / to-dos.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "complete_reminder",
        "description": "Mark a reminder as done, matched by a word or phrase from its text.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "Part of the reminder text to match."}},
            "required": ["query"],
        },
    },
    {
        "name": "control_spotify",
        "description": "Control music playback on Spotify. Use ONLY when the user explicitly "
                       "asks to play/pause/skip music, search for a song/album/artist/playlist, "
                       "set the volume, toggle shuffle or repeat, seek/restart within a track, "
                       "add something to the queue, switch the playback device, or ask what's "
                       "playing. Never for greetings, messages to pass on to people, or general "
                       "conversation.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string",
                           "enum": ["play", "pause", "next", "previous", "search", "volume",
                                    "shuffle", "repeat", "seek", "queue", "play_next",
                                    "transfer", "list_devices", "now_playing"]},
                "query": {"type": "string",
                          "description": "What to play/queue, for action='search', 'queue', or "
                                         "'play_next'. Use 'liked songs' for the user's saved tracks."},
                "kind": {"type": "string", "enum": ["track", "album", "artist", "playlist"],
                         "description": "What to search for when action='search'. "
                                        "Use 'playlist' for mood/genre/theme requests ('lo-fi music', "
                                        "'gaming music', 'chill vibes'). Use 'artist' when the user names "
                                        "a band or musician. Use 'album' for a specific album. "
                                        "Use 'track' for a specific song. Defaults to 'track'."},
                "level": {"type": "integer",
                          "description": "Exact volume 0-100, when action='volume'."},
                "direction": {"type": "string", "enum": ["up", "down"],
                              "description": "Nudge volume up/down when action='volume' and no "
                                             "exact level is given ('louder'→up, 'quieter'→down)."},
                "on": {"type": "boolean",
                       "description": "Turn shuffle on (true) or off (false), when action='shuffle'."},
                "mode": {"type": "string", "enum": ["off", "track", "context"],
                         "description": "Repeat mode, when action='repeat': 'track' repeats the "
                                        "current song, 'context' the album/playlist, 'off' disables."},
                "shuffle": {"type": "boolean",
                            "description": "Start an album/playlist/liked-songs shuffled, with action='search'."},
                "position_seconds": {"type": "integer",
                                     "description": "Absolute position to seek to (0 = restart), action='seek'."},
                "delta_seconds": {"type": "integer",
                                  "description": "Seconds to jump forward (+) or back (-), action='seek'."},
                "device": {"type": "string",
                           "description": "Name (or part) of the device to move playback to, action='transfer'."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "timer_action",
        "description": "Create, cancel, reset, or list countdown timers. Use ONLY for explicit "
                       "timer/countdown requests, not calendar reminders or to-dos. To start "
                       "several timers at once, emit one call per timer.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["start", "cancel", "reset", "list"]},
                "duration_seconds": {"type": "integer",
                                     "description": "Length in seconds, for action 'start' or "
                                                    "'reset' (e.g. 10 minutes = 600, 1m30s = 90)."},
                "label": {"type": "string",
                          "description": "For 'start', an optional name ('pasta', 'tea'). For "
                                         "'cancel'/'reset', picks which timer by name. Never a "
                                         "number word."},
                "all": {"type": "boolean",
                        "description": "For 'cancel'/'reset': apply to every running timer."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "get_system_stats",
        "description": "Report this PC's live CPU, memory, disk, and (if present) GPU usage. Use "
                       "when the user asks how the system/computer is doing or about CPU, RAM, "
                       "memory, GPU, VRAM, disk, or battery. Never for general chit-chat.",
        "parameters": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": ["all", "cpu", "mem", "gpu", "disk", "battery"],
                           "description": "Which figure the user wants; omit or 'all' for a summary."},
            },
            "required": [],
        },
    },
    {
        "name": "launch_app",
        "description": "Open a desktop application by name (e.g. Notepad, Calculator, Settings, "
                       "Chrome, VS Code). Use ONLY when the user asks to open, launch, or start "
                       "a named app — never for opening files, URLs, or messages.",
        "parameters": {
            "type": "object",
            "properties": {
                "app": {"type": "string", "description": "The application name to open."},
            },
            "required": ["app"],
        },
    },
    {
        "name": "daily_briefing",
        "description": "Give the full spoken daily briefing (weather, news and reminders).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]

__all__ = ["TOOLS"]
