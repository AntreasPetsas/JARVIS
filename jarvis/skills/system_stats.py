"""System stats — live CPU / memory / disk / GPU telemetry for the HUD.

CPU/memory/disk/battery come from psutil; GPU (if an NVIDIA card is present) from
a short `nvidia-smi` call. Everything blocking runs in a worker thread so the
event loop is never held up. If psutil isn't installed the skill degrades to a
helpful message rather than crashing the import.
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time

try:
    import psutil
except ImportError:  # keep the app importable without the (Layer-1) dep
    psutil = None  # type: ignore[assignment]

_GB = 1024 ** 3
# Suppress the console window nvidia-smi/subprocess would otherwise flash on Windows.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


async def get_stats() -> dict:
    """Snapshot of system load. Safe to call repeatedly (e.g. for a live panel)."""
    if psutil is None:
        return {"ok": False, "error": "psutil isn't installed. Run: pip install psutil"}
    return await asyncio.to_thread(_snapshot)


def _snapshot() -> dict:
    cpu_percent = psutil.cpu_percent(interval=0.3)  # short blocking sample (in a thread)
    vm = psutil.virtual_memory()
    return {
        "ok": True,
        "cpu": {"percent": round(cpu_percent), "cores": psutil.cpu_count(logical=True)},
        "mem": {"percent": round(vm.percent),
                "used_gb": round(vm.used / _GB, 1), "total_gb": round(vm.total / _GB, 1)},
        "disk": _disk(),
        "gpu": _gpu(),
        "battery": _battery(),
        "uptime": _uptime(),
    }


def _disk() -> dict | None:
    try:
        path = (os.environ.get("SystemDrive", "C:") + os.sep) if os.name == "nt" else "/"
        du = psutil.disk_usage(path)
        return {"percent": round(du.percent),
                "used_gb": round(du.used / _GB, 1), "total_gb": round(du.total / _GB, 1),
                "mount": path.rstrip("\\/") or "/"}
    except Exception:  # noqa: BLE001 — telemetry is best-effort
        return None


def _battery() -> dict | None:
    try:
        bat = psutil.sensors_battery()
    except Exception:  # noqa: BLE001 — not present on desktops / some platforms
        return None
    if bat is None:
        return None
    return {"percent": round(bat.percent), "plugged": bool(bat.power_plugged)}


def _uptime() -> str:
    try:
        secs = int(time.time() - psutil.boot_time())
    except Exception:  # noqa: BLE001
        return ""
    d, rem = divmod(secs, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m}m"
    return f"{m}m"


def _gpu() -> dict | None:
    """NVIDIA GPU load via nvidia-smi. Returns None if there's no card / tool."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, creationflags=_NO_WINDOW,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 5:
        return None
    try:
        name, util, mem_used, mem_total, temp = parts[:5]
        used, total = float(mem_used), float(mem_total)
        return {"name": name, "percent": round(float(util)),
                "mem_used_mb": round(used), "mem_total_mb": round(total),
                "mem_percent": round(used / total * 100) if total else 0,
                "temp": round(float(temp))}
    except ValueError:
        return None


def summary(stats: dict) -> str:
    """One spoken sentence describing current load."""
    if not stats.get("ok"):
        return stats.get("error", "I couldn't read the system stats.")
    cpu = stats["cpu"]["percent"]
    mem = stats["mem"]["percent"]
    gpu = stats.get("gpu")
    parts = [f"CPU at {cpu} percent", f"memory at {mem} percent"]
    if gpu:
        parts.append(f"GPU at {gpu['percent']} percent")
    hot = cpu >= 85 or mem >= 90 or (gpu and gpu["percent"] >= 90)
    lead = "Running a little hot" if hot else "All nominal"
    return f"{lead}, sir — " + ", ".join(parts) + "."


# --- Spoken-query matching (which metric did the user ask about?) ------------------

_METRIC_WORDS = {
    "cpu": (r"cpu", r"processor", r"cores?"),
    "mem": (r"ram", r"memory", r"mem"),
    "gpu": (r"gpu", r"graphics", r"vram", r"video card"),
    "disk": (r"disk", r"storage", r"drive", r"ssd"),
    "battery": (r"battery",),
}
_GENERAL_RE = re.compile(
    r"\b(system stats|telemetry|system status|system load|system info|system performance|"
    r"system resources|resource usage)\b"
    r"|\bhow'?s?\s+(?:is\s+)?(?:the|my)\s+(?:pc|system|computer|machine|rig)\b", re.IGNORECASE)
_QUERY_RE = re.compile(
    r"\b(usage|used|using|load|percent|percentage|level|how|what'?s?|current|currently|temp|"
    r"temperature|much|many|busy|hot|free|tell|show|monitor|running|status|stats?|got|left|"
    r"remaining|report|check)\b", re.IGNORECASE)


def match(text: str) -> list[str] | None:
    """Return the metrics a stats question is asking about, or None if it isn't one.

    Triggers on 'system stats', a metric word plus a query word ('cpu usage', 'how much
    ram'), or a short bare metric ('cpu', 'cpu?').
    """
    low = text.lower()
    if _GENERAL_RE.search(low):
        return ["all"]
    found = [k for k, words in _METRIC_WORDS.items()
             if any(re.search(rf"\b{w}\b", low) for w in words)]
    if not found:
        return None
    wordcount = len(re.findall(r"[a-z0-9]+", low))
    if _QUERY_RE.search(low) or "?" in text or wordcount <= 4:
        return found
    return None


def answer(stats: dict, metrics: list[str]) -> str:
    """Spoken reply targeted at the requested metric(s)."""
    if not stats.get("ok"):
        return stats.get("error", "I couldn't read the system stats, sir.")
    if not metrics or "all" in metrics:
        return summary(stats)
    parts = [_metric_phrase(stats, m) for m in metrics]
    return " ".join(p for p in parts if p) or summary(stats)


def _metric_phrase(stats: dict, metric: str) -> str:
    if metric == "cpu":
        c = stats["cpu"]
        return f"CPU's at {c['percent']} percent across {c['cores']} cores, sir."
    if metric == "mem":
        m = stats["mem"]
        return f"Memory is at {m['percent']} percent — {m['used_gb']} of {m['total_gb']} gigabytes used."
    if metric == "gpu":
        g = stats.get("gpu")
        if not g:
            return "I can't read a dedicated GPU on this machine, sir."
        extra = f", {g['temp']} degrees" if g.get("temp") is not None else ""
        return (f"GPU is at {g['percent']} percent{extra}, with {g['mem_used_mb']} of "
                f"{g['mem_total_mb']} megabytes of VRAM in use.")
    if metric == "disk":
        d = stats.get("disk")
        if not d:
            return "I couldn't read the disk, sir."
        return f"The {d['mount']} drive is {d['percent']} percent full — {d['used_gb']} of {d['total_gb']} gigabytes."
    if metric == "battery":
        b = stats.get("battery")
        if not b:
            return "There's no battery to report on this machine, sir."
        return f"Battery's at {b['percent']} percent, {'charging' if b['plugged'] else 'on battery'}."
    return ""
