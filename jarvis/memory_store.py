"""Persistent conversation history — a swappable store interface + a JSONL backend.

Jarvis keeps a short rolling window of the conversation in memory (see `History` in
`router.py`). This module is what makes that window survive a restart and become
searchable, *without* coupling the rest of the app to a particular database.

Why JSONL? The access pattern is append-on-every-turn, load-recent-on-startup, plus
the occasional keyword search — all at human conversation pace. An append-only JSON
Lines file fits that exactly: O(1) appends (no whole-file rewrite), crash-tolerant
(a torn final line is simply skipped), human-readable / greppable, and stdlib-only.

If a real need for heavy structured / full-text search appears, drop in a SQLite+FTS5
store; for semantic recall, a vector store. Both would implement the same tiny
`ConversationStore` contract below, so nothing else in the app has to change.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Protocol


class ConversationStore(Protocol):
    """Minimal persistence contract for the conversation log."""

    def append(self, role: str, content: str) -> None: ...

    def recent(self, n: int) -> list[dict]: ...

    def search(self, query: str, limit: int = 10) -> list[dict]: ...

    def clear(self) -> None: ...


class JsonlStore:
    """Append-only JSON Lines log: one ``{"ts", "role", "content"}`` object per line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = Lock()

    def append(self, role: str, content: str) -> None:
        content = (content or "").strip()
        if not content:
            return
        line = json.dumps({"ts": time.time(), "role": role, "content": content},
                          ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except json.JSONDecodeError:
                        # Torn / partial line (e.g. a crash mid-write) — skip it.
                        continue
                    if isinstance(rec, dict) and rec.get("role") and rec.get("content"):
                        out.append(rec)
        except OSError:
            return []
        return out

    def recent(self, n: int) -> list[dict]:
        records = self._read_all()
        return records[-n:] if n and n > 0 else records

    def search(self, query: str, limit: int = 10) -> list[dict]:
        q = (query or "").strip().lower()
        if not q:
            return []
        hits = [r for r in self._read_all() if q in (r.get("content") or "").lower()]
        hits.reverse()  # newest first
        return hits[:limit]

    def clear(self) -> None:
        with self._lock:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                try:  # fall back to truncation if the file can't be removed
                    self.path.write_text("", encoding="utf-8")
                except OSError:
                    pass
