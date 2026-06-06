"""Speech-to-text via faster-whisper.

Loads the model once (lazily on first construction). Accepts float32 mono audio
at 16 kHz — the format the mic loop produces.
"""
from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        compute_type = "int8_float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        """audio: float32 mono, 16 kHz, range [-1, 1]. Returns recognised text."""
        if audio is None or audio.size == 0:
            return ""
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=1, vad_filter=True
        )
        return " ".join(s.text for s in segments).strip()
