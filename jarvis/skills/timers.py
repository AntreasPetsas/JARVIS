"""Countdown timers — in-memory, with a live HUD countdown.

A timer fires after N seconds: it drops out of the active list and the router
announces it (`on_fire`). The browser counts down locally from `ends_at`, so we
never have to push a panel every second — one push when a timer starts, cancels,
or fires is enough.

Timers are deliberately not persisted: a half-finished countdown surviving a
restart would be more confusing than useful.
"""
from __future__ import annotations

import asyncio
import re
import time
from threading import Lock
from typing import Awaitable, Callable

_lock = Lock()
_timers: dict[int, dict] = {}  # id -> {id, label, total, ends_at, task}

OnFire = Callable[[dict], Awaitable[None]]


def _public(t: dict) -> dict:
    """JSON-safe view of a timer (drops the asyncio task; adds `remaining`)."""
    return {
        "id": t["id"],
        "label": t["label"],
        "total": t["total"],
        "ends_at": round(t["ends_at"] * 1000),  # epoch ms, for the browser countdown
        "remaining": max(0, round(t["ends_at"] - time.time())),
    }


def list_timers() -> list[dict]:
    with _lock:
        items = sorted(_timers.values(), key=lambda x: x["ends_at"])
    return [_public(t) for t in items]


def start_timer(seconds: int, label: str, on_fire: OnFire) -> dict:
    """Register a timer and schedule it. `on_fire(public_timer)` runs when it elapses."""
    seconds = max(1, int(seconds))
    now = time.time()
    tid = int(now * 1000)
    with _lock:
        while tid in _timers:  # avoid a same-millisecond id collision
            tid += 1
        timer = {"id": tid, "label": (label or "").strip(),
                 "total": seconds, "ends_at": now + seconds}
        _timers[tid] = timer

    async def _run() -> None:
        try:
            await asyncio.sleep(max(0.0, timer["ends_at"] - time.time()))
        except asyncio.CancelledError:
            return
        with _lock:
            _timers.pop(tid, None)
        await on_fire(_public(timer))

    timer["task"] = asyncio.create_task(_run())
    return _public(timer)


def cancel_timer(query: str = "") -> dict | None:
    """Cancel a timer by id or label substring. With no query, cancels the only one."""
    q = (query or "").strip().lower()
    with _lock:
        match = None
        for t in _timers.values():
            if str(t["id"]) == q or (q and q in t["label"].lower()):
                match = t
                break
        if match is None and not q and len(_timers) == 1:
            match = next(iter(_timers.values()))
        if match is None:
            return None
        _timers.pop(match["id"], None)
    task = match.get("task")
    if task is not None:
        task.cancel()
    return _public(match)


def cancel_all() -> int:
    with _lock:
        items = list(_timers.values())
        _timers.clear()
    for t in items:
        task = t.get("task")
        if task is not None:
            task.cancel()
    return len(items)


# --- Helpers shared by the router keyword branch ----------------------------------

_DUR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b",
                     re.IGNORECASE)


def _unit_seconds(unit: str) -> int:
    return {"h": 3600, "m": 60, "s": 1}.get(unit[0].lower(), 0)


def parse_duration(text: str) -> int | None:
    """Sum every '<n> <unit>' pair in the text → total seconds, or None if absent.

    Handles '5 minutes', '1 hour 30 min', '90 seconds', and a few spelled-out cases
    ('an hour', 'half an hour', 'a minute').
    """
    total = 0.0
    found = False
    for m in _DUR_RE.finditer(text):
        total += float(m.group(1)) * _unit_seconds(m.group(2))
        found = True
    if found:
        return int(total)
    # Clock format: M:SS or H:MM:SS (a colon timer like '4:45' = 4 min 45 sec).
    cm = re.search(r"\b(\d{1,2}):([0-5]\d)(?::([0-5]\d))?\b", text)
    if cm:
        a, b, c = cm.group(1), cm.group(2), cm.group(3)
        return int(a) * 3600 + int(b) * 60 + int(c) if c is not None \
            else int(a) * 60 + int(b)
    low = text.lower()
    if "half an hour" in low or "half hour" in low:
        return 1800
    if re.search(r"\b(?:an?|one)\s+hour\b", low):
        return 3600
    if re.search(r"\b(?:a|one)\s+min(?:ute)?\b", low):
        return 60
    return None


def humanize(seconds: int) -> str:
    """'5 minutes', '1 hour 30 minutes', '45 seconds' for a duration in seconds."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h} hour{'s' if h != 1 else ''}")
    if m:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    if s and not h:  # drop seconds once we're into hours — it's just noise
        parts.append(f"{s} second{'s' if s != 1 else ''}")
    return " ".join(parts) or "0 seconds"
