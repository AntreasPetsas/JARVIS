"""Spotify control — open the app and play/pause via Windows media keys.

v1 uses the global media-key virtual codes, which Spotify Desktop honors when
it owns the active media session. No OAuth needed. The Spotify Web API can be
layered on later for richer control (specific tracks, volume, devices).
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import time

# Virtual-key codes for media keys
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
KEYEVENTF_KEYUP = 0x0002


def _tap(vk: int) -> None:
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.keybd_event(vk, 0, 0, 0)
    time.sleep(0.05)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


def open_spotify(exe_path: str = "") -> dict:
    try:
        if exe_path and os.path.exists(exe_path):
            subprocess.Popen([exe_path])
        else:
            os.startfile("spotify:")  # type: ignore[attr-defined]  # launches desktop app if installed
        return {"ok": True, "message": "Opening Spotify."}
    except OSError:
        os.startfile("https://open.spotify.com")  # type: ignore[attr-defined]
        return {"ok": True, "message": "Opening Spotify in your browser."}


def play_pause() -> dict:
    _tap(VK_MEDIA_PLAY_PAUSE)
    return {"ok": True, "message": "Toggled playback."}


def next_track() -> dict:
    _tap(VK_MEDIA_NEXT_TRACK)
    return {"ok": True, "message": "Skipping to the next track."}


def prev_track() -> dict:
    _tap(VK_MEDIA_PREV_TRACK)
    return {"ok": True, "message": "Going to the previous track."}
