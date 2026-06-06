"""Configuration loading for Jarvis.

Merges config.yaml (or config.example.yaml as a fallback) over built-in
defaults, and loads secrets from .env. Use dotted lookups: cfg.get('llm.model').
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

_DEFAULTS: dict[str, Any] = {
    "location": {"city": "London", "units": "metric"},
    "assistant": {
        "name": "Jarvis",
        "wake_word": "hey_jarvis",
        "wake_sensitivity": 0.5,
        "voice": "en-GB-RyanNeural",
    },
    "llm": {
        "provider": "ollama",
        "model": "qwen2.5:7b-instruct",
        "host": "http://localhost:11434",
        "max_tokens": 500,
        "tools": True,  # let the model pick skills itself (tool-calling) for off-keyword requests
        "history_turns": 6,  # rolling conversation memory: how many user/assistant exchanges to keep
    },
    "news": {
        "software": {
            "enabled": True,
            "sources": ["hackernews", "devto"],
            "reddit_subs": ["programming"],
            "limit": 6,
        },
        "hobbies": [],
    },
    "spotify": {
        "exe_path": "",
        "redirect_uri": "http://127.0.0.1:8765/spotify/callback",
        "default_device": "",
    },
    "briefing": {
        "on_startup": True,
        "sections": ["greeting", "weather", "software_news", "hobby_news", "reminders"],
    },
    "voice": {
        "enabled": True,         # run the mic loop (wake word + speech-to-text)
        "tts": True,             # speak replies with edge-tts (browser plays the audio)
        "input_device": None,    # None = system default mic; or an index/name
        "stt_model": "base",     # whisper size: tiny | base | small
        "stt_device": "cpu",     # cpu | cuda
        "silence_rms": 0.015,    # endpointing: frame RMS below this counts as silence
        "max_command_seconds": 7,
        "wake_debug": True,      # print "Hey Jarvis" scores to the server console (set False once tuned)
    },
    "server": {"host": "127.0.0.1", "port": 8765, "open_browser": True},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    def __init__(self, data: dict[str, Any]):
        self._d = data

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def get(self, path: str, default: Any = None) -> Any:
        """Dotted-path lookup, e.g. cfg.get('llm.model', 'default')."""
        cur: Any = self._d
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    @property
    def data(self) -> dict[str, Any]:
        return self._d

    @staticmethod
    def env(key: str, default: str | None = None) -> str | None:
        return os.environ.get(key, default)


def load_config() -> Config:
    load_dotenv(ROOT / ".env")
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        cfg_path = ROOT / "config.example.yaml"
    user: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    return Config(_deep_merge(_DEFAULTS, user))
