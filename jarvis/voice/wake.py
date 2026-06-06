"""Wake-word detection via openWakeWord (onnx backend, ships a 'hey_jarvis' model)."""
from __future__ import annotations

import numpy as np
from openwakeword.model import Model


class WakeWord:
    def __init__(self, name: str = "hey_jarvis", threshold: float = 0.5):
        self.name = name
        self.threshold = threshold
        # onnx avoids the tflite-runtime dependency (which lacks Windows/3.13 wheels).
        self.model = Model(wakeword_models=[name], inference_framework="onnx")

    def detect(self, frame_int16: np.ndarray) -> float:
        """frame_int16: int16 mono 16 kHz, ~1280 samples (80 ms). Returns the score."""
        preds = self.model.predict(frame_int16)
        # Key may be the bare name or the model filename stem — take the best match.
        if self.name in preds:
            return float(preds[self.name])
        return float(max(preds.values())) if preds else 0.0

    def reset(self) -> None:
        try:
            self.model.reset()
        except AttributeError:
            pass
