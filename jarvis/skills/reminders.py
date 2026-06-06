"""Reminders / to-do — a simple JSON-backed list."""
from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock

DATA = Path(__file__).resolve().parent.parent / "data"
STORE = DATA / "reminders.json"
_lock = Lock()


def _load() -> list[dict]:
    if not STORE.exists():
        return []
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save(items: list[dict]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(items, indent=2), encoding="utf-8")


def list_reminders(include_done: bool = False) -> list[dict]:
    items = _load()
    return items if include_done else [r for r in items if not r.get("done")]


def add_reminder(text: str) -> dict:
    with _lock:
        items = _load()
        item = {
            "id": int(time.time() * 1000),
            "text": text,
            "done": False,
            "created": time.strftime("%Y-%m-%d %H:%M"),
        }
        items.append(item)
        _save(items)
    return item


def complete_reminder(query: str) -> dict | None:
    """Mark the first reminder matching `query` (by id or substring) as done."""
    with _lock:
        items = _load()
        match = None
        for r in items:
            if str(r["id"]) == query or query.lower() in r["text"].lower():
                r["done"] = True
                match = r
                break
        _save(items)
    return match


def clear_done() -> int:
    with _lock:
        items = _load()
        kept = [r for r in items if not r.get("done")]
        removed = len(items) - len(kept)
        _save(kept)
    return removed
