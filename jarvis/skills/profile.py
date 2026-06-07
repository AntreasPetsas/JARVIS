"""The user profile — durable, human-editable facts Jarvis knows about you.

Stored as a Markdown file (``jarvis/data/profile.md``) so you can open and edit it by
hand. Jarvis injects it into the LLM prompt so replies are personalised, and the
"get to know me" interview (see ``onboarding.py``) fills it in. Mirrors the simple
file-store pattern in ``reminders.py``.

The file is deliberately a transparent, hand-editable format:

    # What Jarvis knows about you
    <!-- Edit this file freely. Jarvis reads it to personalise replies. -->

    - **Name:** Andreas
    - **Based in:** Athens

    ## Notes
    - prefers tea over coffee
"""
from __future__ import annotations

import re
from pathlib import Path
from threading import Lock

DATA = Path(__file__).resolve().parent.parent / "data"
STORE = DATA / "profile.md"
_lock = Lock()

_HEADER = "# What Jarvis knows about you"
_NOTE = "<!-- Edit this file freely. Jarvis reads it to personalise replies. -->"
_NOTES_HEADER = "## Notes"

# A structured fact bullet:  - **Key:** value   (colon may sit inside or after the bold)
_FACT_RE = re.compile(r"^\s*-\s*\*\*(?P<key>[^*]+?)\*\*\s*:?\s*(?P<value>.+?)\s*$")


def profile_text() -> str:
    """Raw Markdown contents (or '' if there's no profile yet)."""
    if not STORE.exists():
        return ""
    try:
        return STORE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def facts() -> dict[str, str]:
    """Parse '- **Key:** value' bullets into an insertion-ordered dict."""
    out: dict[str, str] = {}
    for line in profile_text().splitlines():
        m = _FACT_RE.match(line)
        if m:
            key = m.group("key").strip().rstrip(":").strip()
            val = m.group("value").strip()
            if key and val:
                out[key] = val
    return out


def is_empty() -> bool:
    return profile_text() == ""


def _ensure_scaffold(text: str) -> list[str]:
    """Return the file's lines, guaranteeing the header + note are present."""
    lines = text.splitlines() if text else []
    if not any(ln.strip() == _HEADER for ln in lines):
        lines = [_HEADER, _NOTE, ""] + lines
    return lines


def _write(lines: list[str]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    STORE.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def set_fact(key: str, value: str) -> None:
    """Upsert a structured fact bullet by key, preserving the rest of the file."""
    key = (key or "").strip()
    value = (value or "").strip()
    if not key or not value:
        return
    with _lock:
        lines = _ensure_scaffold(profile_text())
        bullet = f"- **{key}:** {value}"
        for i, line in enumerate(lines):
            m = _FACT_RE.match(line)
            if m and m.group("key").strip().rstrip(":").strip().lower() == key.lower():
                lines[i] = bullet
                _write(lines)
                return
        # New key — insert just after the last existing fact bullet (keeps facts grouped).
        insert_at = len(lines)
        for i, line in enumerate(lines):
            if _FACT_RE.match(line):
                insert_at = i + 1
        lines.insert(insert_at, bullet)
        _write(lines)


def add_note(text: str) -> None:
    """Append a free-form note under a '## Notes' section (accumulates, no overwrite)."""
    text = (text or "").strip().rstrip(".")
    if not text:
        return
    with _lock:
        lines = _ensure_scaffold(profile_text())
        if not any(ln.strip() == _NOTES_HEADER for ln in lines):
            lines += ["", _NOTES_HEADER]
        lines.append(f"- {text}")
        _write(lines)


# Note: there is intentionally no clear/delete function — Jarvis never forgets you.
# Facts are only added or updated; the file at data/profile.md is hand-editable if you
# ever want to remove something yourself.
