"""App launcher — open desktop apps by name. Windows-first, pure stdlib.

Resolution order for a spoken name:
  1. user aliases from config (`app_launcher.aliases`)
  2. a small built-in map of common Windows apps
  3. anything on PATH (`shutil.which`)

A spec of the form ``uri:<target>`` is launched with ``os.startfile`` (shell
protocols / Store apps, e.g. ``ms-settings:``); anything else is treated as an
executable and started with ``subprocess.Popen``.

`launch()` returns ``{ok, resolved, message}``. `resolved` is False when the name
matched nothing — the router uses that to fall through to the LLM, so "open up to
me" stays conversation instead of being treated as a launch.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

# Spoken name -> launch spec. "uri:" => os.startfile; otherwise an exe on PATH.
_BUILTINS: dict[str, str] = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe", "calc": "calc.exe",
    "paint": "mspaint.exe",
    "explorer": "explorer.exe", "file explorer": "explorer.exe", "files": "explorer.exe",
    "task manager": "taskmgr.exe",
    "command prompt": "cmd.exe", "cmd": "cmd.exe",
    "terminal": "wt.exe", "windows terminal": "wt.exe",
    "powershell": "powershell.exe",
    "control panel": "control.exe",
    "settings": "uri:ms-settings:",
    "snipping tool": "snippingtool.exe", "snip": "uri:ms-screenclip:",
    "camera": "uri:microsoft.windows.camera:",
    "calendar": "uri:outlookcal:", "mail": "uri:outlookmail:",
    "store": "uri:ms-windows-store:", "microsoft store": "uri:ms-windows-store:",
    "maps": "uri:bingmaps:",
}

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _norm(name: str) -> str:
    return " ".join((name or "").lower().split())


def _resolve(name: str, aliases: dict) -> str | None:
    """Return a launch spec ('uri:...' or an exe path/name), or None if unknown."""
    key = _norm(name)
    if not key:
        return None

    norm_aliases = {_norm(k): str(v) for k, v in (aliases or {}).items()}
    if key in norm_aliases:
        return norm_aliases[key]
    if key in _BUILTINS:
        return _BUILTINS[key]
    # loose alias match ("open my code editor" -> alias "code editor")
    for k, v in norm_aliases.items():
        if k and (k in key or key in k):
            return v

    # On PATH, by full name then first word ("google chrome" -> "chrome").
    for cand in (key, key.split()[0]):
        hit = shutil.which(cand) or shutil.which(cand + ".exe")
        if hit:
            return hit
    return None


def _start_uri(target: str) -> None:
    startfile = getattr(os, "startfile", None)
    if startfile is None:  # non-Windows
        raise OSError("shell launch is only supported on Windows")
    startfile(target)


def launch(name: str, aliases: dict | None = None) -> dict:
    spec = _resolve(name, aliases or {})
    label = (name or "").strip().title() or "that"
    if spec is None:
        return {"ok": False, "resolved": False,
                "message": f"I don't know how to open '{(name or '').strip()}'. "
                           "Add it under app_launcher.aliases in config.yaml."}
    try:
        if spec.startswith("uri:"):
            _start_uri(spec[4:])
        elif os.path.isfile(spec):
            subprocess.Popen([spec], creationflags=_NO_WINDOW)
        else:  # a bare exe name on PATH
            subprocess.Popen([spec], creationflags=_NO_WINDOW)
        return {"ok": True, "resolved": True, "message": f"Opening {label}."}
    except (OSError, ValueError) as e:
        return {"ok": False, "resolved": True,
                "message": f"I couldn't open {label} ({e})."}


# Filler words trimmed off a spoken target ("open notepad please" -> "notepad").
_TRAIL_RE = re.compile(r"\s+(?:please|for me|now|app|application|window|program)$",
                       re.IGNORECASE)


def clean_target(raw: str) -> str:
    return _TRAIL_RE.sub("", (raw or "").strip(" .?!,'\"")).strip()
