"""The microphone loop: wake word -> record -> transcribe -> hand to the router.

Runs in a background thread (sounddevice is blocking). Audio never leaves the
machine and is only transcribed *after* the wake word fires (or a manual
push-to-talk trigger). Results are pushed back into the asyncio loop via the
two callbacks supplied by the server.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Awaitable, Callable

import numpy as np

from ..config import Config

SAMPLE_RATE = 16000
FRAME = 1280  # 80 ms @ 16 kHz — openWakeWord's expected chunk size

EventCb = Callable[[dict], Awaitable[None]]    # broadcast a message to the HUD(s)
CommandCb = Callable[[str], Awaitable[None]]   # run a transcribed command


class VoiceAssistant:
    def __init__(self, cfg: Config, loop: asyncio.AbstractEventLoop,
                 on_event: EventCb, on_command: CommandCb):
        self.cfg = cfg
        self.loop = loop
        self.on_event = on_event
        self.on_command = on_command
        self.threshold = float(cfg.get("assistant.wake_sensitivity", 0.5))
        self.wake_name = cfg.get("assistant.wake_word", "hey_jarvis")
        self.silence_rms = float(cfg.get("voice.silence_rms", 0.015))
        self.max_cmd = float(cfg.get("voice.max_command_seconds", 7))
        self.device = cfg.get("voice.input_device", None)
        self._stop = threading.Event()
        self._listen_now = threading.Event()  # manual push-to-talk
        self._mute_until = 0.0                # wake paused until this time (auto-expires)
        self._busy = False
        self.wake_debug = bool(cfg.get("voice.wake_debug", False))
        self._last_dbg = 0.0
        self._thread: threading.Thread | None = None
        self._wake = None
        self._stt = None

    # ---- lifecycle ----
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="voice", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def trigger_listen(self) -> None:
        """Push-to-talk: capture one command without the wake word."""
        self._listen_now.set()

    def mute(self, seconds: float | None = None) -> None:
        """Pause wake detection while a reply plays. Auto-expires so it can't stick on."""
        self._mute_until = time.time() + min((seconds or 12.0) + 1.0, 30.0)

    def unmute(self) -> None:
        self._mute_until = 0.0

    # ---- thread -> event loop bridges ----
    def _emit(self, msg: dict) -> None:
        asyncio.run_coroutine_threadsafe(self.on_event(msg), self.loop)

    def _dispatch(self, text: str) -> None:
        asyncio.run_coroutine_threadsafe(self.on_command(text), self.loop)

    # ---- the loop ----
    def _run(self) -> None:
        import sounddevice as sd

        from .stt import Transcriber
        from .wake import WakeWord

        try:
            self._wake = WakeWord(self.wake_name, self.threshold)
            self._stt = Transcriber(self.cfg.get("voice.stt_model", "base"),
                                    self.cfg.get("voice.stt_device", "cpu"))
        except Exception as e:  # noqa: BLE001
            self._emit({"type": "voice_status", "ok": False, "error": f"voice init failed: {e}"})
            return

        try:
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                    blocksize=FRAME, device=self.device)
            stream.start()
        except Exception as e:  # noqa: BLE001
            self._emit({"type": "voice_status", "ok": False, "error": f"microphone error: {e}"})
            return

        self._emit({"type": "voice_status", "ok": True,
                    "message": f"Voice online — say '{self.wake_name.replace('_', ' ')}'."})
        try:
            while not self._stop.is_set():
                frame = stream.read(FRAME)[0].reshape(-1)
                if self._busy:
                    continue
                triggered = self._listen_now.is_set()
                if not triggered:
                    if time.time() < self._mute_until:   # a reply is playing; skip our own audio
                        continue
                    score = self._wake.detect(frame)
                    if self.wake_debug and score > 0.1 and time.time() - self._last_dbg > 0.25:
                        self._last_dbg = time.time()
                        print(f"[wake] score={score:.3f}  (threshold={self.threshold})")
                    triggered = score >= self.threshold
                if triggered:
                    self._listen_now.clear()
                    self._handle_utterance(stream)
        finally:
            stream.stop()
            stream.close()

    def _handle_utterance(self, stream) -> None:
        self._busy = True
        try:
            self._emit({"type": "state", "state": "listening"})
            audio = self._record_command(stream)
            self._emit({"type": "state", "state": "thinking"})
            text = self._stt.transcribe(audio).strip()
            if len(text) >= 2:
                self._dispatch(text)
            else:
                self._emit({"type": "subtitle", "text": "I didn't quite catch that, sir."})
                self._emit({"type": "state", "state": "idle"})
        except Exception as e:  # noqa: BLE001
            self._emit({"type": "voice_status", "ok": False, "error": str(e)})
            self._emit({"type": "state", "state": "idle"})
        finally:
            time.sleep(0.3)  # cooldown so we don't catch the tail of our own reply
            if self._wake:
                self._wake.reset()
            self._busy = False

    def _record_command(self, stream) -> np.ndarray:
        """Record from wake until ~0.8 s of trailing silence, capped at max_cmd."""
        frames: list[np.ndarray] = []
        frame_sec = FRAME / SAMPLE_RATE
        hang, silent_for, elapsed, spoke = 0.8, 0.0, 0.0, False
        while elapsed < self.max_cmd and not self._stop.is_set():
            frame = stream.read(FRAME)[0].reshape(-1)
            frames.append(frame)
            rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))
            if rms >= self.silence_rms:
                spoke, silent_for = True, 0.0
            else:
                silent_for += frame_sec
            elapsed += frame_sec
            if spoke and silent_for >= hang and elapsed > 0.5:
                break
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames).astype(np.float32) / 32768.0
