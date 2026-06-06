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
                "level": {"type": "integer",
                          "description": "Volume 0-100, when action='volume'."},
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
        "name": "daily_briefing",
        "description": "Give the full spoken daily briefing (weather, news and reminders).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]

__all__ = ["TOOLS"]
