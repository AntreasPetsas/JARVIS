"""The "get to know me" interview — a short scripted Q&A that seeds the profile.

State lives in an `InterviewState` instance (one per server, created in `create_app`
and shared by typed + voice input). The router drives it: while an interview is
active, every user turn is treated as the answer to the current question until the
script finishes or the user bows out. Questions are *scripted* (not model-generated),
so the interview works even with no LLM available — the model, when present, is only
used to tidy answers into concise profile values.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import Config

# Each step fills one profile key with the answer to a spoken question.
DEFAULT_QUESTIONS: list[dict] = [
    {"key": "Name", "q": "What should I call you?"},
    {"key": "Based in", "q": "Where are you based?"},
    {"key": "Work", "q": "What do you do for work?"},
    {"key": "Working on", "q": "What are you working on at the moment?"},
    {"key": "Interests", "q": "What do you do for fun — any hobbies or interests?"},
    {"key": "Preferred tone",
     "q": "And how would you like me to talk to you — formal, casual, or full butler?"},
]

_START_RE = re.compile(
    r"\b(start getting to know me|getting to know me|get to know me|"
    r"set ?up my profile|onboard me|do the interview|let'?s do the interview|"
    r"interview me|ask me (?:some |a few )?questions)\b",
    re.IGNORECASE,
)
_SKIP_RE = re.compile(r"^\s*(?:skip|pass|skip (?:it|that|this))\s*\.?\s*$", re.IGNORECASE)
_STOP_RE = re.compile(
    r"\b(stop|cancel|never ?mind|that'?s enough|that is enough|no more|"
    r"we'?re done|i'?m done|quit|enough)\b",
    re.IGNORECASE,
)


def is_start(text: str) -> bool:
    return bool(_START_RE.search(text or ""))


def is_skip(text: str) -> bool:
    return bool(_SKIP_RE.match(text or ""))


def is_stop(text: str) -> bool:
    return bool(_STOP_RE.search(text or ""))


def load_questions(cfg: Config) -> list[dict]:
    """Use `memory.interview` from config if provided, else the default script."""
    raw = cfg.get("memory.interview", None)
    out: list[dict] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("key") and item.get("q"):
                out.append({"key": str(item["key"]), "q": str(item["q"])})
    return out or list(DEFAULT_QUESTIONS)


@dataclass
class InterviewState:
    """Mutable per-server interview progress. `active` is False when idle."""

    active: bool = False
    index: int = 0
    answered: int = 0
    clean: bool = False  # whether to LLM-tidy answers (decided once, at start)
    questions: list[dict] = field(default_factory=lambda: list(DEFAULT_QUESTIONS))

    def current(self) -> dict | None:
        if self.active and 0 <= self.index < len(self.questions):
            return self.questions[self.index]
        return None

    def reset(self) -> None:
        self.active = False
        self.index = 0
        self.answered = 0
        self.clean = False
