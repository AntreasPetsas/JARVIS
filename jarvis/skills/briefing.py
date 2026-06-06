"""Daily briefing — gather weather, news and reminders, then (optionally) have
the LLM turn them into a short spoken summary with an activity suggestion.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

from ..config import Config
from ..llm import get_llm
from . import news, reminders, weather

SYSTEM = (
    "You are Jarvis, a concise, witty British AI butler. Given the user's daily data, "
    "speak a short briefing (max ~120 words): greet briefly, summarise the weather and "
    "suggest one fitting activity, highlight the 2 most interesting headlines, and mention "
    "any reminders. Natural spoken style — no markdown, no bullet lists."
)


def _greeting(name: str) -> str:
    h = datetime.now().hour
    part = "morning" if h < 12 else "afternoon" if h < 18 else "evening"
    return f"Good {part}. {name} online."


async def gather_briefing(cfg: Config) -> dict:
    """Collect all briefing data concurrently. Shaped for the HUD panels."""
    city = cfg.get("location.city", "London")
    units = cfg.get("location.units", "metric")

    async def safe(coro, fallback):
        try:
            return await coro
        except Exception as e:  # noqa: BLE001 — one dead source must not sink the briefing
            return fallback(e)

    wx, sw, hb = await asyncio.gather(
        safe(weather.get_weather(city, units), lambda e: {"ok": False, "error": str(e)}),
        safe(news.get_software_news(cfg.get("news.software", {})), lambda e: []),
        safe(news.get_hobby_news(cfg.get("news.hobbies", [])), lambda e: []),
    )
    return {
        "greeting": _greeting(cfg.get("assistant.name", "Jarvis")),
        "weather": wx,
        "software_news": sw,
        "hobby_news": hb,
        "reminders": reminders.list_reminders(),
    }


def _format_for_llm(data: dict) -> str:
    lines: list[str] = []
    wx = data.get("weather", {})
    if wx.get("ok"):
        lines.append(
            f"Weather in {wx['city']}: {wx['condition']}, {wx['temp']}{wx['unit']} "
            f"(feels {wx['feels_like']}{wx['unit']}), high {wx['high']} / low {wx['low']}, "
            f"{wx.get('precip_chance', 0)}% chance of precipitation."
        )
    for label, key in (("Software headlines", "software_news"), ("Hobby headlines", "hobby_news")):
        if data.get(key):
            lines.append(f"{label}:")
            lines += [f"- {a['title']} ({a['source']})" for a in data[key][:5]]
    if data.get("reminders"):
        lines.append("Reminders: " + "; ".join(r["text"] for r in data["reminders"][:5]))
    return "\n".join(lines)


async def spoken_briefing(cfg: Config, data: dict | None = None) -> dict:
    if data is None:
        data = await gather_briefing(cfg)
    llm = get_llm(cfg)
    text = ""
    if await llm.available():
        try:
            text = await llm.complete(SYSTEM, _format_for_llm(data), cfg.get("llm.max_tokens", 500))
        except Exception:  # noqa: BLE001
            text = ""
    if not text:
        # Graceful fallback when no LLM is configured/reachable yet.
        wx = data.get("weather", {})
        bits = [data.get("greeting", "Hello.")]
        if wx.get("ok"):
            bits.append(f"It's {wx['temp']}{wx['unit']} and {wx['condition'].lower()} in {wx['city']}.")
        news_bits = []
        if data.get("software_news"):
            news_bits.append(f"{len(data['software_news'])} software stories")
        if data.get("hobby_news"):
            news_bits.append(f"{len(data['hobby_news'])} from your hobbies")
        if news_bits:
            bits.append("I've pulled " + " and ".join(news_bits) + ".")
        open_rem = [r for r in data.get("reminders", []) if not r.get("done")]
        if open_rem:
            n = len(open_rem)
            bits.append(f"You have {n} reminder{'s' if n != 1 else ''}.")
        text = " ".join(bits)
    return {"text": text, "data": data}
